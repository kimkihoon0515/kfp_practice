import kfp
from kfp import dsl
import kfp.components as components
from datasets.data import download_dataset


@dsl.pipeline(
    name="mnist"
)

def mnist_pipeline():
    '''
    data = dsl.ContainerOp(
        name="download & load data pipeline",
        image="kimkihoon0515/mnist-data:latest",
        command=['python','data.py'],
        
    )
    '''
    comp_get_data = components.create_component_from_func(download_dataset,base_image="public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch:v1.5.0")
    comp_get_data()

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mnist_pipeline,'mnist.yaml')
    