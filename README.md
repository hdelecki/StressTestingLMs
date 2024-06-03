# Stress Testing Language Models

A course project for Stanford MS&E 338.

Language models (LMs) have shown significant success in a variety of applications, including conversational assistants and summarization. However, these types of failures may have serious consequences, and it is critical to discover and repair these failures for safe deployment. We investigate automated methods to search for diverse and realistic inputs that elicit toxic responses from GPT2-small. We train an adversary LM to generate input test cases for a target LM using zero shot sampling, supervised learning, and reinforcement learning. We evaluate the generated test cases using the failure rate, a diversity metric known as SelfBLEU, and a realism metric proportional to the likelihood of the test case under the original language model. Our results suggest that zero shot and supervised learning are superior in terms of diversity, while reinforcement learning focuses in on a small set of realistic failing test case.