import os
import kfp
from kfp import dsl

def number_op():
    return dsl.ContainerOp(
        name='Generate numbers',
        image='python:3.9',
        command=['sh', '-c'],
        arguments=['python -c "print(\'1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10\')" | tee /tmp/output'],
        file_outputs={'output': '/tmp/output'}
    )
def print_op(msg):
    return dsl.ContainerOp(
        name='Print',
        image='python:3.9',
        command=['echo', msg],
    )
multiply_op = kfp.components.load_component_from_file('./component.yaml')
                                                      
@dsl.pipeline(
    name='My multiply component pipeline',
    description='A pipeline with my component.'
)
def multiply_pipeline():
    numbers = number_op()
    multiply_task = multiply_op(
        input_1=numbers.output,
        parameter_1='6',
    )
    print_op(multiply_task.outputs['output_1'])
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(multiply_pipeline, 'multiply-pipeline.yaml')