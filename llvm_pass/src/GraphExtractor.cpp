/**
 * @file GraphExtractor.cpp
 * @brief LLVM ModulePass for extracting Control-Data-Flow Graphs (CDFG) from IR.
 *
 *
 * Key features:
 * - Control Flow Extraction (Basic Block connectivity)
 * - Data Flow Extraction (SSA Def-Use chains)
 * - Type Hierarchy (Explicit nodes for types)
 * - Memory Dependence Analysis (Alias Analysis)
 *
 * @author xaiqo
 * @version 1.0.0
 */

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

    /**
     * @class GraphExtractor
     * @brief Main pass class that extracts graph structure from LLVM Module.
     */
    struct GraphExtractor : public ModulePass {
        static char ID; // Pass identification

        GraphExtractor() : ModulePass(ID) {}

        // --- Data Structures ---

        /**
         * @brief Represents a node in the extracted graph.
         */
        struct Node {
            std::string id;         ///< Unique identifier (memory address)
            std::string label;      ///< Human-readable label (e.g., "add", "i32")
            std::string category;   ///< Node type: "Instruction", "Value", "Type"
            std::string data_type;  ///< Return type ID (for instructions/values)
        };

        /**
         * @brief Represents an edge in the extracted graph.
         */
        struct Edge {
            std::string src;        ///< Source Node ID
            std::string dst;        ///< Destination Node ID
            std::string type;       ///< Edge Type: "Control", "Data", "Type", "Memory"
        };

        // Graph storage
        std::vector<Node> nodes;
        std::vector<Edge> edges;
        std::set<std::string> visited_types; // To avoid duplicate type nodes

        /**
         * @brief Declares analysis dependencies.
         * We require AAResultsWrapperPass for Memory Dependence Analysis.
         */
        void getAnalysisUsage(AnalysisUsage &AU) const override {
            AU.setPreservesAll();
            AU.addRequired<AAResultsWrapperPass>();
        }

        /**
         * @brief Main entry point for the pass.
         * Iterates over all functions in the module and builds the graph.
         */
        bool runOnModule(Module &M) override {
            nodes.clear();
            edges.clear();
            visited_types.clear();

            for (Function &F : M) {
                // Skip external declarations (e.g. printf)
                if (F.isDeclaration()) continue;

                // Retrieve Alias Analysis results for the current function
                AAResults &AA = getAnalysis<AAResultsWrapperPass>(F).getAAResults();

                std::vector<Instruction*> memory_insts;

                for (BasicBlock &BB : F) {
                    processBasicBlock(BB);
                    
                    // Collect memory instructions (Load/Store) for batch analysis
                    for (Instruction &I : BB) {
                         if (isa<LoadInst>(&I) || isa<StoreInst>(&I)) {
                             memory_insts.push_back(&I);
                         }
                    }
                }
                
                // Perform O(N^2) pairwise alias analysis on memory instructions
                // to detect implicit data dependencies through memory.
                analyzeMemoryDependence(AA, memory_insts);
            }

            emitJSON();
            return false; // We do not modify the IR
        }

    private:
        /**
         * @brief Analyzes memory dependencies between Load/Store instructions.
         * Adds "Memory_Alias" or "Memory_MustAlias" edges to the graph.
         * 
         * @param AA Reference to AliasAnalysis results.
         * @param insts Vector of Load/Store instructions in the function.
         */
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

        /**
         * @brief Generates a unique string ID for an LLVM Value based on its pointer address.
         */
        std::string getValueID(Value *v) {
            std::stringstream ss;
            ss << "v_" << (const void*)v;
            return ss.str();
        }

        /**
         * @brief Generates a unique string ID for an LLVM Type.
         */
        std::string getTypeID(Type *t) {
            std::string str;
            raw_string_ostream os(str);
            t->print(os);
            return "t_" + os.str(); // e.g. "t_i32" or "t_i32*"
        }

        /**
         * @brief Adds a node representing a Type to the graph if not already present.
         */
        void addTypeNode(Type *t) {
            std::string tid = getTypeID(t);
            if (visited_types.count(tid)) return;
            visited_types.insert(tid);

            std::string type_str;
            raw_string_ostream os(type_str);
            t->print(os);

            nodes.push_back({tid, os.str(), "Type", ""});

            // TODO: Recursively handle composite types (Structs, Arrays)
            // for a finer-grained type graph.
        }

        /**
         * @brief Processes a BasicBlock to extract instructions and local CFG/DFG.
         */
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
                    
                    // Create nodes for non-instruction operands (Arguments, Constants)
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
         * @brief Serializes the graph to JSON and prints to stderr.
         * We use stderr to separate graph output from compilation logs.
         */
        void emitJSON() {
            errs() << "{\n";
            errs() << "  \"nodes\": [\n";
            for (size_t i = 0; i < nodes.size(); ++i) {
                errs() << "    {";
                errs() << "\"id\": \"" << nodes[i].id << "\", ";
                errs() << "\"label\": \"" << escapeJson(nodes[i].label) << "\", ";
                errs() << "\"category\": \"" << nodes[i].category << "\"";
                errs() << "}";
                if (i < nodes.size() - 1) errs() << ",";
                errs() << "\n";
            }
            errs() << "  ],\n";
            errs() << "  \"edges\": [\n";
            for (size_t i = 0; i < edges.size(); ++i) {
                errs() << "    {";
                errs() << "\"src\": \"" << edges[i].src << "\", ";
                errs() << "\"dst\": \"" << edges[i].dst << "\", ";
                errs() << "\"type\": \"" << edges[i].type << "\"";
                errs() << "}";
                if (i < edges.size() - 1) errs() << ",";
                errs() << "\n";
            }
            errs() << "  ]\n";
            errs() << "}\n";
        }

        /**
         * @brief Helper to escape special characters for valid JSON.
         */
        std::string escapeJson(const std::string &s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                if (c == '\\') out += "\\\\";
                else if (c == '"') out += "\\\"";
                else if (c == '\n') out += "\\n";
                else if (c == '\t') out += "\\t";
                else out += c;
            }
            return out;
        }
    };
}

// Pass registration
char GraphExtractor::ID = 0;
static RegisterPass<GraphExtractor> X("extract-graph", "LLVM IR to Graph Extractor");
