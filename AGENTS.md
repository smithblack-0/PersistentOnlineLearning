# Repository instructions for coding and research agents

For substantial research, experiment design, engineering, review, or documentation work, read [`documents/llms/README.md`](documents/llms/README.md) before acting.

[`STATUS.md`](STATUS.md) is shared project documentation for humans and tools. It describes the latest durable position, but it is not an autonomous work queue and does not establish the user's current objective.

General project procedures live under [`documents/procedures/`](documents/procedures/). LLM-specific warnings and routing live under [`documents/llms/`](documents/llms/). Do not move general procedures back into the LLM folder.

Do not run substantive model training in an agent's local container. Local work may include implementation, unit and contract tests, deterministic smoke validation, static analysis, and preparation of runnable Colab jobs. GPU experiments and scientific training runs belong in the user's execution environment.

Do not merge pull requests, modify CI or workflow files, expose runtime credentials, or change an accepted central scientific interpretation without authorization. When the user grants broad design authority, make provisional scientific and architectural choices, test and critique them, and report them afterward rather than interrupting at every ordinary fork.
