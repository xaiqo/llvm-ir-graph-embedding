#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"
#include <set>
#include <vector>
#include <string>
#include <sstream>

using namespace llvm;

namespace {

    struct GraphExtractor : public ModulePass {
        static char ID;
        GraphExtractor() : ModulePass(ID) {}

        struct Node {
            std::string id;
            std::string label; // Opcode or Type name
            std::string category; // "Instruction", "Value", "Type"
            std::string data_type; // Return type of instruction
        };

        struct Edge {
            std::string src;
            std::string dst;
            std::string type; // "Control", "Data", "Type", "Memory"
        };

        std::vector<Node> nodes;
        std::vector<Edge> edges;
        std::set<std::string> visited_types;

        // --- Analysis Declaration ---
        void getAnalysisUsage(AnalysisUsage &AU) const override {
            AU.setPreservesAll();
            AU.addRequired<AAResultsWrapperPass>();
        }

        // --- Main Entry Point ---
        bool runOnModule(Module &M) override {
            nodes.clear();
            edges.clear();
            visited_types.clear();

            for (Function &F : M) {
                if (F.isDeclaration()) continue;

                // Get Alias Analysis results for this function
                AAResults &AA = getAnalysis<AAResultsWrapperPass>(F).getAAResults();

                std::vector<Instruction*> memory_insts;

                for (BasicBlock &BB : F) {
                    processBasicBlock(BB);
                    
                    // Collect memory instructions for AA
                    for (Instruction &I : BB) {
                         if (isa<LoadInst>(&I) || isa<StoreInst>(&I)) {
                             memory_insts.push_back(&I);
                         }
                    }
                }
                
                // Perform Alias Analysis (Pairwise)
                analyzeMemoryDependence(AA, memory_insts);
            }

            printJSON();
            return false;
        }

        void analyzeMemoryDependence(AAResults &AA, const std::vector<Instruction*> &insts) {
            for (size_t i = 0; i < insts.size(); ++i) {
                for (size_t j = i + 1; j < insts.size(); ++j) {
                    Instruction *I1 = insts[i];
                    Instruction *I2 = insts[j];

                    // We only care if at least one is a Store (Write)
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

        // --- Helpers ---

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

            // Recurse for contained types (e.g. pointer to int, struct fields)
            if (t->isPointerTy()) {

            } 
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
                    
                    // Ensure the operand has a node (might be a Constant or Argument)
                    if (isa<Argument>(v) || isa<Constant>(v)) {

                         bool exists = false; 
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

        void printJSON() {  
            errs() << "{\n";
            errs() << "  \"nodes\": [\n";
            for (size_t i = 0; i < nodes.size(); ++i) {
                errs() << "    {\"id\": \"" << nodes[i].id 
                       << "\", \"label\": \"" << nodes[i].label 
                       << "\", \"category\": \"" << nodes[i].category << "\"}";
                if (i < nodes.size() - 1) errs() << ",";
                errs() << "\n";
            }
            errs() << "  ],\n";
            errs() << "  \"edges\": [\n";
            for (size_t i = 0; i < edges.size(); ++i) {
                errs() << "    {\"src\": \"" << edges[i].src 
                       << "\", \"dst\": \"" << edges[i].dst 
                       << "\", \"type\": \"" << edges[i].type << "\"}";
                if (i < edges.size() - 1) errs() << ",";
                errs() << "\n";
            }
            errs() << "  ]\n";
            errs() << "}\n";
        }
    };
}

char GraphExtractor::ID = 0;
static RegisterPass<GraphExtractor> X("extract-graph", "LLVM IR to Graph Extractor");

