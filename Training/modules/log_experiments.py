
import mlflow
import time

class MlFlow:
    def __init__(self,experiment_name) -> None:
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name = experiment_name)
        
    def log_result(self,parameters,name,metrics):

        metrics, IoUs, dets, tresholds  = metrics


        print("IoUs:",IoUs, "dets:",dets)

        print(len(tresholds),len(IoUs),len(dets))

        with mlflow.start_run(run_name = name ):
            
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
   
            for idx in range(len(IoUs)):
                #IoUs, dets, tresholds
                seuil = int(round(tresholds[idx]*100, 0))  # use high number to avoid several points on the same step
                mlflow.log_metrics({'IoU':       IoUs[idx], 
                                    'det':       dets[idx],
                                    "tresholds": tresholds[idx]}, 
                                                 step=seuil)