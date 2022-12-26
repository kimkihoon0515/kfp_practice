import kfp
import kfp.components as comp
from kfp import dsl

def print_op(msg):
    
    return dsl.ContainerOp(
        name='Print',
        command=['ehco',msg],
    )



@dsl.pipeline(
    name='iris-train',
    description='training iris-dataset demo'
)

def pipeline():
    add_p = dsl.ContainerOp(
        name="load iris data pipeline",
        arguments=[
            '--data_path', './Iris.csv'
        ],
        file_outputs={'iris' : '/iris.csv'}
    )

    ml = dsl.ContainerOp(
        name="training pipeline",
        arguments=[
            '--data', add_p.outputs['iris']
        ]
    )

    ml.after(add_p)
    baseline = 0.7
    
    with dsl.Condition(ml.outputs['accuracy'] > baseline) as check_condition:
        print_op(f"accuracy는 {ml.outputs['accuracy']}로 accuracy baseline인 {baseline}보다 크다.")
    
    with dsl.Condition(ml.outputs['accuracy'] < baseline) as check_condition:
        print_op(f"accuracy는 {ml.outputs['accuracy']}로 accuracy baseline인 {baseline}보다 작다.")

    
if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline, __file__ + ".tar.gz")