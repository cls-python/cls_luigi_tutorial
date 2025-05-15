import luigi
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta


class TaskA(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return luigi.LocalTarget("output/taskA_output.txt")

    def run(self):
        with self.output().open('w') as f:
            f.write("Task A completed")

class TaskB(luigi.Task, LuigiCombinator):
    abstract = False
    task_a = ClsParameter(tpe=TaskA.return_type())

    def requires(self):
        return self.task_a()

    def output(self):
        return luigi.LocalTarget("output/taskB_output.txt")

    def run(self):
        with self.input().open() as input_file, self.output().open('w') as output_file:
            data = input_file.read()
            output_file.write("Task B completed with input: " + data)

if __name__ == '__main__':

    target = TaskB.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if not actual is None or actual == 0:
        max_results = actual
    validator = RepoMeta.get_unique_abstract_task_validator()
    results = [t() for t in inhabitation_result.evaluated[0:max_results]
               if validator.validate(t())]
    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        no_schedule_error = luigi.build(results, local_scheduler=True)
    else:
        print("No results!")
