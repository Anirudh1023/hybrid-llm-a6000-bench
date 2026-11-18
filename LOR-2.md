To the Admissions Committee,

I am pleased to recommend Anirudh Bocha for admission to your graduate program. As his manager at Samsung Research, I have worked with him for [duration] and observed his contributions to on-device machine learning across inference and language-model training. Anirudh combines strong engineering skills with an ability to understand the research objectives behind a problem. He is particularly effective when a promising idea depends on several components working together reliably.

I first saw these strengths in his work on our on-device inference runtime, which supports execution across CPU, GPU, and NPU backends. Different vendor runtimes and evolving framework versions created model conversion, compatibility, and deployment challenges. Anirudh helped migrate updates from LiteRT into our runtime, resolved issues in integrating CPU and GPU execution paths, and worked with product teams to diagnose model-specific problems. He also developed and improved an automated GPU caching tool that standardized cache generation and made a recurring deployment workflow easier to use. What stood out was his willingness to follow a problem across framework boundaries until the resulting system worked for the teams relying on it.

Anirudh brought the same approach to NNTrainer, Samsung’s open-source framework, where he worked to make language-model inference more practical on mobile CPUs. He implemented optimizations for normalization operations and drove the integration of 4-bit quantization, connecting weight quantization with inference-engine changes and optimized hardware kernels. He also added support for the LFM2 architecture. This work required more than integrating individual components: he had to understand how a model’s architecture, numerical representation, and execution path affected one another. His contributions strengthened the framework’s ability to support low-memory LLM inference and new model architectures.

A particularly demanding example of his engineering ownership came during our collaboration with Samsung Research Korea on on-device LLM personalization. Once memory-efficient training became feasible, running it quickly on mobile hardware remained a challenge. Anirudh investigated why moving computation to the NPU did not by itself deliver the expected gains and identified synchronization and data movement between the CPU and NPU as significant bottlenecks. He designed an asynchronous dispatch mechanism using shared memory to improve coordination between the processors. The resulting training execution was eight times faster than CPU-only execution, and the design has been filed for a patent.

This project also called for coordination beyond any one implementation. Anirudh worked with team members to integrate independently developed components, diagnose failures across them, and bring the training pipeline toward a functional end-to-end system. He understood that a research method becomes useful only when its underlying execution is reliable enough to evaluate. His attention to both the technical detail and the wider objective made him a dependable contributor to a complex collaborative effort.

Across these projects, I have seen Anirudh approach unfamiliar problems with initiative and persistence. He profiles a system to understand where it falls short, considers the implications of a change across components, and follows through on integration and validation. I recommend him strongly for graduate study and am confident that his engineering judgment and commitment to building usable systems will enable him to contribute to research.

Sincerely,

Thummala Pallavi  
Manager, SRIB  
Samsung Research  
[Contact information]
