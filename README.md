# Structure

## Architecture

### Senses (Input)

Recognize speech, images, and other sensor inputs.

* Speech recognition
* Image/face recognition

### Knowledge (Core)

Manage AI logic, time, personality, memory, and context.

* Access via configuration profile
  * Separate memory
  * context
  * personality layers
    * Behavioral changes
* Memory and context
  * Short term
  * long term
  * Personality
* ML Model
* External data
  * Time

### Communication (Output)

Respond via speech, control visuals, interact with games or environments.

* Text-to-speech
* VTuber avatar manipulation
* Game integration

## Unsolved issues

### Imitating Thinking About an Issue (e.g., "The answer is a... No, rather b")

To make the model emulate a process of self-reflection, where it changes its initial answer (e.g., "The answer is A... No, rather B"):

1. Prompting Style:
   * Use longer prompts that encourage self-debate. For example:
   * "Consider the pros and cons of A. Think carefully before concluding."
   * LLama3’s structure can support this by prompting it to "think out loud."
2. Chain-of-Thought Prompting:
   * LLama3 is good with step-by-step reasoning. You can guide the model to reconsider its conclusions by asking it to reflect: "Now that you've thought about A, is there something you're missing? Reevaluate."

### Storing Facts / Committing Them to Memory

To have the model commit facts to memory and decide what to store:

* Knowledge Core as Authority:
  Use the LLM to process information and suggest what should be stored as fact, but have a dedicated   memory service decide what is committed. The LLM should output candidates for facts: "I just learned   that X. Should I remember this?"
  Then, based on criteria or rules in the knowledge module, you decide what is persisted.
* Automatic Memory Commitment:
  After each conversation or task, feed the LLM-generated conclusions into the memory database (e.g., Redis/SQLite). Example workflow:
  LLM identifies important facts → Sends to a “fact manager” API → The fact manager validates and persists in memory.
* Feedback and Decision:
  Let the model suggest multiple potential interpretations of the input and store only the final "decision": "It seems like we decided X is correct, let me save that."
  Keep logs in the memory service to track committed decisions for future context.

### Summary Approach for LLama3:

1. Chain-of-Thought prompting: Encourage the LLM to reason through problems.
2. Memory Commitment: Have the LLM suggest facts but use an external service to validate and store them.
3. Personality Context: Adjust the prompts based on the personality and make decisions accordingly.

## Plan

### Development phases

1. Basic MVP with simple input/output
   1. Speech recognition -> Text processing -> TTS response.
   2. Simple visual manipulations (e.g., mouth movement).
2. Add memory/personality
   1. Track previous conversations, context-aware responses.
   2. Improve facial expressions and gestures.
3. Integrate with external systems (games, streams).
   1. Game/streaming overlay, dynamic interactions.