from azureml.core import Run

run = Run.get_context()
run.log('message','hello world!')

run.complete()
