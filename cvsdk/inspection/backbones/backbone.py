import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from cvsdk.inspection.backbones.classifier import Classifier


class BackboneModel():
    def __init__(self, filepath=None):
        # Classification layer of model
        self.name = None
        self.model = None
        self.filepath = filepath
        self.classifier = None
        self.classifier_input_size = None
        self.feature_extractor = None
        pass
    
    def load_weights(self):
        if self.filepath:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            options = torch.load(self.filepath, map_location=torch.device(device))
            self.load_state_dict(options)
            
    def train(self):
        self.model.train()
        return self
            
    def eval(self):
        self.model.eval()
        return self
    
    def zero_grad(self):
        self.model.zero_grad()
    
    def parameters(self):
        return self.model.parameters()
    
    def get_features(self, input_batch, key):
        """
        Extract feature tensors from model
        
        :param input_batch: input tensor (image encoded as tensor)
        :param key: feature layer key
        """
        if isinstance (key, str):
            key = int(key)
        if isinstance(key, int):
            key = self.feature_keys[key]
        with torch.no_grad():
            features = self.feature_extractor(input_batch)
            features = features[key]
            return features
    
    def get_nodes(self):
        train_nodes, eval_nodes = get_graph_node_names(self.model)
        return train_nodes, eval_nodes
    
    def classify_tensor(self, feature_tensor):
        """
        Run feature tensor through classification layer of model
        """
        return self.classifier(feature_tensor)
    
    def set_classifier(self, classifier):
        """
        Replace the classifier of the model
        """
        # This method needs to be implemented in the child classes
        pass
    
    def create_classifier_network(self, options):
        """
        Create a classification layer
        """
        # Creating the Feedforward Classifier
        classifier = Classifier(input_size = self.classifier_input_size,
                             output_size = options['output_size'],
                             hidden_layers = options['hidden_layers'], 
                             drop_p = options['drop_p'])
        
        return classifier
        
    def replace_classifier(self, options):
        """
        Replace the classification layer
        """
        cls = self.create_classifier_network(options)
        self.set_classifier(self.create_classifier_network(options))
        
    def load_state_dict(self, options):
        """
        Load weights of a trained model instance
        """
        self.replace_classifier(options)
        self.model.load_state_dict(options['state_dict'])
        
    def requires_grad(b: bool):
        """
        Pass True if you want to train this network, false otherwise
        :param b: True if layers require gradient, False otherwise
        """
        for param in self.model.parameters():
            param.requires_grad = b
