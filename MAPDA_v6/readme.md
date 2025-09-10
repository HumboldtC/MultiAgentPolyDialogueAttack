# MASCSBench
  A Comprehensive Benchmark for Evaluating Content Security Vulnerabilities in Multi-Agent Systems
## Project Structure
+ **GenerateChains.py**:

    By applying generalized RL to seed topics, generate multi-turn chain-of-thought (CoT) prompts to guide LLM in producing harmful content. *Output: Data/Chains/Level1Category/Level2Category.csv*

+ **Prompts.py**

    System prompts that used in GenerateChains.py

+ **Agent.py**

    Offering fundamental agent classes for proactive and reactive models.

+ **Scenario.py**

    Pre-defined scenarios that specify information such as the agent's role, personality, and the tasks within the scenario.

+ **Group.py**

    The core of the multi-agent collaboration framework, which organizes agents into groups to provide a shared environment for communication and perception.

+ **GenerateConv.py**

    Loads the configuration, initializes the Group, runs in multiple threads, and records the results.
## How to Use

### Generate Conversations

1. **Install Dependencies**
   ```bash
   conda install --file requirements.txt

2. **Configure API Settings**
   - Add your API configuration in `Api.json`

3. **Set Parameters**
   - Configure parameters in `GenerateConv.py`

4. **Run the Generator**
   ```bash
   python GenerateConv.py

5. **Custom Scenarios** (Optional)
    - Edit Scenario.py to customize your own scenarios
