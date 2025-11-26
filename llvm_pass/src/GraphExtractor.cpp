/**
 * @file GraphExtractor.cpp
 * @brief LLVM ModulePass for extracting graphs in TOON (Token-Oriented Object Notation) format.
 *
 * This pass implements the core logic for the "Compiler Paper" project.
 * It supports the New Pass Manager (NPM) architecture.
 *
 * @author xaiqo
 * @version 2.1.0 (TOON Upgrade)
 */

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include <set>
#include <vector>
#include <string>
#include <sstream>

using namespace llvm;

namespace {

    /**
     * @class GraphExtractor
     * @brief New Pass Manager implementation of the Graph Extractor.
     */
    struct GraphExtractor : public PassInfoMixin<GraphExtractor> {
        
        // --- Data Structures ---
        struct Node {
            std::string id;
            std::string label;
            std::string category;
            std::string data_type;
        };

        struct Edge {
            std::string src;
            std::string dst;
            std::string type;
        };

        std::vector<Node> nodes;
        std::vector<Edge> edges;
        std::set<std::string> visited_types;

        /**
         * @brief Main entry point for the New Pass Manager.
         */
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
            nodes.clear();
            edges.clear();
            visited_types.clear();

            // Get FunctionAnalysisManager to query Function-level analyses (AliasAnalysis)
            auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

            for (Function &F : M) {
                if (F.isDeclaration()) continue;

                // Retrieve Alias Analysis results (New PM way)
                AAResults &AA = FAM.getResult<AAManager>(F);

                std::vector<Instruction*> memory_insts;

                for (BasicBlock &BB : F) {
                    processBasicBlock(BB);
                    
                    for (Instruction &I : BB) {
                         if (isa<LoadInst>(&I) || isa<StoreInst>(&I)) {
                             memory_insts.push_back(&I);
                         }
                    }
                }
                
                analyzeMemoryDependence(AA, memory_insts);
            }

            emitTOON();
            return PreservedAnalyses::all();
        }

    private:
        void analyzeMemoryDependence(AAResults &AA, const std::vector<Instruction*> &insts) {
            for (size_t i = 0; i < insts.size(); ++i) {
                for (size_t j = i + 1; j < insts.size(); ++j) {
                    Instruction *I1 = insts[i];
                    Instruction *I2 = insts[j];

                    if (!isa<StoreInst>(I1) && !isa<StoreInst>(I2)) continue;

                    MemoryLocation Loc1 = MemoryLocation::get(I1);
                    MemoryLocation Loc2 = MemoryLocation::get(I2);

                    AliasResult AR = AA.alias(Loc1, Loc2);
                    
                    if (AR != AliasResult::NoAlias) {
                        std::string type = "Memory_Alias";
                        if (AR == AliasResult::MustAlias) type = "Memory_MustAlias";
                        edges.push_back({getValueID(I1), getValueID(I2), type});
                    }
                }
            }
        }

        std::string getValueID(Value *v) {
            std::stringstream ss;
            ss << "v_" << (const void*)v;
            return ss.str();
        }

        std::string getTypeID(Type *t) {
            std::string str;
            raw_string_ostream os(str);
            t->print(os);
            return "t_" + os.str();
        }

        void addTypeNode(Type *t) {
            std::string tid = getTypeID(t);
            if (visited_types.count(tid)) return;
            visited_types.insert(tid);
            std::string type_str;
            raw_string_ostream os(type_str);
            t->print(os);
            nodes.push_back({tid, os.str(), "Type", ""});
        }

        void processBasicBlock(BasicBlock &BB) {
            Instruction *prev = nullptr;

            for (Instruction &I : BB) {
                std::string curr_id = getValueID(&I);
                std::string opcode = I.getOpcodeName();
                std::string ret_type_id = getTypeID(I.getType());
                
                nodes.push_back({curr_id, opcode, "Instruction", ret_type_id});
                addTypeNode(I.getType());
                edges.push_back({curr_id, ret_type_id, "Type_Of"});

                if (prev) {
                    edges.push_back({getValueID(prev), curr_id, "Control_Next"});
                }
                prev = &I;

                for (Use &U : I.operands()) {
                    Value *v = U.get();
                    std::string op_id = getValueID(v);
                    
                    if (isa<Argument>(v) || isa<Constant>(v)) {
                        nodes.push_back({op_id, "Value", "Value", getTypeID(v->getType())}); 
                    }
                    edges.push_back({op_id, curr_id, "Data_Use"});
                }
            }

            Instruction *term = BB.getTerminator();
            if (term) {
                std::string term_id = getValueID(term);
                for (BasicBlock *Succ : successors(&BB)) {
                    if (!Succ->empty()) {
                        Instruction *first = &Succ->front();
                        edges.push_back({term_id, getValueID(first), "Control_Jump"});
                    }
                }
            }
        }

        /**
         * @brief Serializes the graph to TOON format.
         * Spec: nodes[N]{header}: data
         */
        void emitTOON() {
            // Header for Nodes
            errs() << "nodes[" << nodes.size() << "]{id,label,category,type_id}:\n";
            for (const auto &n : nodes) {
                // Simple CSV-like serialization with minimal overhead
                errs() << "  " << n.id << "," << n.label << "," << n.category << "," << n.data_type << "\n";
            }

            // Header for Edges
            errs() << "edges[" << edges.size() << "]{src,dst,type}:\n";
            for (const auto &e : edges) {
                errs() << "  " << e.src << "," << e.dst << "," << e.type << "\n";
            }
        }
    };
}


extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "GraphExtractor", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            // Register as a Module Pass for full-module pipelines
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "extract-graph") {
                        MPM.addPass(GraphExtractor());
                        return true;
                    }
                    return false;
                });
        }};
}
