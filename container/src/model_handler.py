from collections import namedtuple
import json
import os

from dgl.data.chem import smiles_to_bigraph, BaseAtomFeaturizer, CanonicalAtomFeaturizer
from dgl import DGLGraph
import torch

def smiles_to_dgl_graph(smiles_str: str, node_featurizer: BaseAtomFeaturizer = CanonicalAtomFeaturizer()) -> DGLGraph:
    return smiles_to_bigraph(smiles_str, atom_featurizer=node_featurizer)

def logit2probability(input_logit: torch.Tensor) -> torch.Tensor:
    odds = torch.exp(input_logit)
    prob = odds/ (1 + odds)
    return prob

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.model = None
        self.shapes = None

    def get_input_data_shapes(self, model_dir, checkpoint_prefix):
        """
        Get the model input data shapes and return the list

        :param model_dir: Path to the directory with model artifacts
        :param checkpoint_prefix: Model files prefix name
        :return: prefix string for model artifact files
        """
        shapes_file_path = os.path.join(model_dir, "{}-{}".format(checkpoint_prefix, "shapes.json"))
        if not os.path.isfile(shapes_file_path):
            raise RuntimeError("Missing {} file.".format(shapes_file_path))

        with open(shapes_file_path) as f:
            self.shapes = json.load(f)

        data_shapes = []

        for input_data in self.shapes:
            data_name = input_data["name"]
            data_shape = input_data["shape"]
            data_shapes.append((data_name, tuple(data_shape)))

        return data_shapes

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        print('MODEL DIR {}'.format(model_dir))

        # Load model
        try:
            self.model = torch.load(os.path.join(model_dir, 'gnn_model.pt'))
            print('Model successfully loaded...')
        except RuntimeError:
            raise

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        atom_data_field = 'h'
        
        smiles = [json.loads(r['body'].decode('utf-8')) for r in request]
        print('INPUT SMILES {}'.format(smiles))
        
        graphs = [smiles_to_dgl_graph(s) for s in smiles]
        feats = [g.ndata.pop(atom_data_field) for g in graphs]
        print('Input successfully preprocessed...')
        return zip(graphs, feats)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        
        probs = []
        for graph, feat in model_input:
            logit = self.model(graph, feat)
            prob = logit2probability(logit)
            # format output so it returns list[float,...]
            prob = prob[0][0]
            probs.append(prob.detach().numpy().tolist())

        print('Inference successfully completed...')
        return probs

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        output = [str(inference_output)]
        print('Postprocess successfully completed...')
        return output

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

