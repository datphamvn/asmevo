# **Research Project: LLM-guided Evolutionary Assembly Optimization**

## **1\. Motivation and Research Questions**

This project investigates an LLM-based agentic approach to compiler optimization for software engineering by optimizing compiler-generated assembly code. The LLM Multi-Agent System (MAS) as a variation operator in evolutionary search, rewriting x86-64 assembly produced by gcc \-O3. The goal is to evaluate search effectiveness and reliability, beyond demonstrating an LLM API call.

* **RQ1:** Can evolutionary search produce executables faster than the \-O3 baseline?  
* **RQ2:** Do population-based search and selection outperform independent best-of-N sampling and single-trajectory iterative refinement under the same budget?  
* **RQ3:** Do improvements hold on inputs excluded from the search? Which assembly and ABI errors occur most frequently?

**Scope:** A Python package, Linux x86-64, one fixed compiler/version, and one fixed LLM. Modify assembly only, preserving interfaces, ABI, and program behavior. LLM fine-tuning is outside this project's scope. Expected contributions are the search loop, reproducible evaluation infrastructure, and empirical analysis.

## **2\. Related Work and Benchmarks**

### **2.1. Related Work**

**LLM-based Assembly Optimization**

* **Stochastic Superoptimization**  
  * [Paper](https://arxiv.org/abs/1211.0557), [code](https://github.com/StanfordPL/stoke)  
  * Classic stochastic superoptimization of assembly. A reference for search and correctness checking. It uses stochastic search rather than an LLM.  
* **SuperCoder: Assembly Program Superoptimization with Large Language Models**  
  * [Paper](https://arxiv.org/abs/2505.11480), [code](https://github.com/Anjiang-Wei/SuperCoder)  
  * LLM-based optimization of x86-64 assembly compiled with GCC \-O3. A reference for best-of-N and iterative refinement. Its RL method differs from the proposed evolutionary loop.  
* **Agentic Code Optimization via Compiler-LLM Cooperation**  
  * [Paper](https://arxiv.org/abs/2604.04238), [code](https://github.com/amazon-science/acclaim)  
  * Coordinates source/IR/assembly optimization with testing. A reference for orchestration and feedback.

**LLM-based Evolutionary Search and Agent Workflows**

These methods inform the search design, not all use multi-agent systems.

* **FunSearch**  
  * [Paper](https://www.nature.com/articles/s41586-023-06924-6), [code](https://github.com/google-deepmind/funsearch)  
  * Combines LLM-generated program variations with automated evaluation and evolutionary selection. A reference for maintaining a population of useful programs and using evaluation feedback.  
* **EoH: Evolution of Heuristics**  
  * [Paper](https://arxiv.org/abs/2401.02051), [code](https://github.com/FeiLiu36/EoH)  
  * Evolves natural-language descriptions and code together to design heuristics. A reference for parent selection and prompting strategies that generate new candidates.  
* **HSEvo**  
  * [Paper](https://arxiv.org/abs/2412.14995), [code](https://github.com/datphamvn/HSEvo)  
  * Combines LLM-based evolutionary program search with harmony search and diversity management. A reference for balancing exploration and convergence.  
* **MCTS-AHD**  
  * [Paper](https://arxiv.org/abs/2501.08603), [code](https://github.com/zz1358m/MCTS-AHD-master)  
  * Uses Monte Carlo Tree Search to explore LLM-generated heuristics. A reference for allocating search effort across promising candidates and an alternative to population-based selection.

### **2.2. Benchmarks**

* **SuperCoder:** The primary benchmark, containing CodeNet C programs compiled to x86-64 assembly with GCC \-O3 and accompanied by test cases. [Dataset](https://huggingface.co/datasets/LLM4Code/llm_superoptimizer_ds)  
* **Hacker’s Delight tasks in STOKE:** Small bit-manipulation programs used in superoptimization research. An optional supplementary benchmark or fallback. [Code and examples](https://github.com/StanfordPL/stoke)

## 

## **3\. Agentic Evolutionary Loop**

**Proposed loop:**

1. **Initialize:** Build the \-O3 baseline, test and time it, and add it to the population/archive.  
2. **Select:** Use tournament selection to choose parents from a population of up to 4 valid candidates.  
3. **Mutate:** Provide the LLM with parent assembly, ABI constraints, and feedback. Generate 4 offspring per generation. LLM-based crossover using two parents is an optional extension, rather than arbitrary instruction concatenation.  
4. **Evaluate:** Assemble/link → differential tests → runtime measurement. Exclude incorrect, crashing, or timed-out candidates from selection. Retain errors as feedback.  
5. **Survive:** Retain the baseline and the best valid candidates, up to 4 members. Remove duplicates by hash. Record parent IDs, generation, prompts, and outcomes.  
6. **Repeat/stop:** Run 3 generations with at most 12 candidate generations per run, including repairs and failed candidates. Check the selected candidate on held-out inputs. Fall back to the baseline if correctness checks fail or improvement cannot be confirmed.

Fitness for a test-passing candidate is T\_baseline / T\_candidate. Invalid candidates receive \-inf. Use one LLM with build/test/measurement tools. An agentic loop does not require multiple LLM agents.

**Protocol:** Compare \-O3, best-of-12, 12-step refinement, and 3×4 evolution. Use the same model, benchmarks, per-call token limits, and maximum budget of 12 candidate generations for the search methods. Run 3 search seeds per method and record actual token use and cost. Finalize prompts/configurations using the development set. Warm up and interleave repeated measurements on the same CPU. Report median runtime, variability, geometric-mean speedup, pass/improvement rates, and cost. Score fallback executables as 1× while reporting all failures separately. Final evaluation inputs must not enter prompts or selection.

## **4\. Assignments & Timeline**

**Every member follows the same four stages:** background study, coding, experiments and report preparation. Each member picks individual tasks from the shared checklist and adds their email as PIC. **Every member must own at least 3 coding tasks, and every task must have an owner.**

### **4.1. Background Study: 05/09–19/09**

During this period, the team has been studying the foundations and reviewing the skills needed for the project:

* Compiler optimization overview: Compilation stages, compiler flags, pass ordering, intermediate representations, and assembly optimization, with a focus on direct assembly rewriting.  
* Literature survey: LLM-based compiler optimization, superoptimization, related methods, benchmarks, and evaluation practices.  
* Agent workflows: Tool use, iterative feedback, and build–test–measure loops for generating and evaluating candidate programs.  
* LLM-based evolutionary computation (EC): Populations, parent selection, LLM-guided mutation, optional crossover, fitness evaluation, and survivor selection.  
* Project skills review: Python modular design, Git/GitHub collaboration, Linux and compiler toolchains, x86-64 assembly and ABI basics, testing, experimental analysis, and LaTeX reporting.

### **4.2. Coding Tasks: 19/09–01/10**

Each member picks at least **3 coding tasks** and makes at least **3 code commits by 01/10**. Fixes during experiments count. Assign every task and preserve individual commit history.

* Initialize the codebase (first commit on 19/09). PIC: \_\_\_\_\_\_\_\_\_\_  
* Define schemas and interfaces. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement budget tracking. PIC: \_\_\_\_\_\_\_\_\_\_  
* Add lineage and budget tests, and analyze search traces. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement the loader and manifest. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement dataset splits. PIC: \_\_\_\_\_\_\_\_\_\_  
* Build the GCC baseline for at least 2 tasks. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement the LLM client. PIC: huynhvancuong20@gmail.com  
* Implement mutation prompts and feedback. PIC: huynhvancuong20@gmail.com  
* Implement the population model and candidate mutation. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement selection, elitism, and deduplication. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement an isolated runner. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement timeouts, resource limits, and correctness checks. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement best-of-N. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement iterative refinement. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement the experiment runner and run 3 seeds per method. PIC: \_\_\_\_\_\_\_\_\_\_  
* Implement plots and analyze results. PIC: huynhvancuong20@gmail.com

### **4.3. Experiments and Ablations: 27/09–01/10 (TBD)**

- Baseline 1  
- Baseline 2  
- Our method  
- Ablations  
  - Ablations 1  
  - Ablations 2  
  - Ablations 3

### **4.4. Report and Presentation: 01/10–04/10**

Each member picks one unassigned report section below and includes their experiment and ablation findings.

* Architecture, persistence, and reproducibility. PIC: \_\_\_\_\_\_\_\_\_\_  
* Benchmarks, dataset preparation, and experimental setup. PIC: \_\_\_\_\_\_\_\_\_\_  
* Motivation, research questions, related work, and LLM generation. PIC: \_\_\_\_\_\_\_\_\_\_  
* Evolutionary selection, validation, and failure analysis. PIC: \_\_\_\_\_\_\_\_\_\_  
* Baselines, aggregate results, and plots. Assemble the report. PIC: \_\_\_\_\_\_\_\_\_\_

## **5\. Delivery Checklist and Notes**

- [ ] Github repo  
- [ ] Report follows the course template: motivation, related work, method, setup, results, threats to validity, and contributions. LaTeX preferred.  
- [ ] Final check & Email

**Notes:**

* Submit the project and email the lecturer by **October 5, 2026**, 10 days before the target exam on **October 15, 2026**. Changes after the submission deadline will not be assessed.  
* **The project accounts for 80% of the grade and the individual oral interview accounts for 20%.**

