from src.workers.worker import TaskType, WorkerType

LeanTaskType = TaskType("lean")
CompletionTaskType = TaskType("completion")
PolicyValueTaskType = TaskType("policy_value")
PolicyValuePostProcessTaskType = TaskType("policy_value_post_process")

KillTaskType = TaskType("kill")

LeanWorkerType = WorkerType("lean", (LeanTaskType, KillTaskType))
MCTSWorkerType = WorkerType(
    "mcts", (LeanTaskType, CompletionTaskType, PolicyValueTaskType, KillTaskType))
LinearInferenceWorkerType = WorkerType(
    "linear_inference", (LeanTaskType, CompletionTaskType, PolicyValueTaskType, KillTaskType))

CompletionWorkerType = WorkerType(
    "completion", (CompletionTaskType, KillTaskType))
ContextWorkerType = WorkerType("context", (PolicyValueTaskType, KillTaskType))
PolicyValueWorkerType = WorkerType(
    "policy_value", (PolicyValuePostProcessTaskType, KillTaskType))
FastPolicyValueWorkerType = WorkerType(
    "fast_policy_value", (PolicyValueTaskType, KillTaskType))
