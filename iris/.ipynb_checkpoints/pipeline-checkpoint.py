import kfp
import kfp.components as comp
from kfp import dsl
@dsl.pipeline(
    name='kf-iris',
    description='kubflow-pipeline iris test'
)

def kf_iris_pipeline():
    add_p = dsl.ContainerOp(
        name="load iris data pipeline",
        image="kimkihoon0515/kf_iris_preprocessing:0.5",
        arguments=[
            '--data_path', './Iris.csv'
        ],
        file_outputs={'iris' : '/iris.csv'}
    )

    ml = dsl.ContainerOp(
        name="training pipeline",
        image="kimkihoon0515/kf_iris_train:0.5",
        arguments=[
            '--data', add_p.outputs['iris']
        ]
    )

    ml.after(add_p)
    
if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(kf_iris_pipeline, "pipeline.yaml")