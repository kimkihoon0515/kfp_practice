import kfp
from kfp import dsl
@kfp.dsl.component
def train_component_op():
    return kfp.dsl.ContainerOp(
        name='mnist-train',
        image='kangwoo/kfp-mnist:kfp'
    )
@dsl.pipeline(
    name='My pipeline',
    description='My machine learning pipeline'
)
def my_pipeline():
    train_task = train_component_op()
if __name__ == '__main__':
    # Compile
    pipeline_package_path = 'mnist-pipeline.yaml'
    kfp.compiler.Compiler().compile(my_pipeline, pipeline_package_path)