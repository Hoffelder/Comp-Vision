import json
import itertools
import warnings
warnings.filterwarnings("ignore")

class GridSeach:
    def __init__(self) -> None:
        f = open('hiper_parameters/data.json')
        self.data = json.load(f)

    def generate_hiper_parameters_list(self,model_name) ->(list,list,list):

        hiper_parameters = list()
        parameters_names = list() 

        for parameter_name in self.data[model_name]:

            parameters_names.append( parameter_name )
            hiper_parameters.append( tuple(self.data[model_name][parameter_name]))
        
        return model_name,parameters_names,list(itertools.product(*hiper_parameters))
    
    def iterate_on_all_parameters(self):
        for model_name in self.data:
            yield self.generate_hiper_parameters_list(model_name)

    def iterate_by_name(self,model_name):
        model_name_,parameters_names_, hiper_parameters_ = self.generate_hiper_parameters_list(model_name)
        
        for parameter_values in hiper_parameters_:

            

            test_case = {name:param for name,param in zip(parameters_names_,parameter_values)}
                
            yield test_case


    def generate_parameter(self):
        for model_name,parameters_names, hiper_parameters in self.iterate_on_all_parameters():
            for parameter_values in hiper_parameters:
                test_case = {name:param for name,param in zip(parameters_names,parameter_values)} 
                
                yield model_name,test_case


