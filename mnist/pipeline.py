import kfp
from kfp import dsl


@dsl.pipeline(
    name="mnist"
)

def mnist_pipeline():
    data = dsl.ContainerOp(
        name="download & load data pipeline",
        image="kimkihoon0515/mnist-data:latest",
        command=['python','data.py'],
        
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mnist_pipeline,'mnist.yaml')