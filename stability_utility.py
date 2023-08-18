import shap
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from scipy.special import binom
from scipy.stats import rankdata
import random
import time
from itertools import combinations


def num_samples(num_layer, nbF):
    """Number of instances of a given layer"""

    num_subset_sizes = np.int(np.ceil((nbF - 1) / 2.0))

    if num_layer > num_subset_sizes : 
        num_layer = num_subset_sizes
    
    
    if num_layer == 1 :
        nsubsets = 2 * binom(nbF,1)

    elif num_layer == num_subset_sizes :
        nsubsets = 2**nbF - 2

    else : 
        k = 0
        nsubsets = 0
        while k < num_layer :
            k = k+1
            nsubsets += 2 * binom(nbF,k)
            

    return np.int(nsubsets)

def compute_vsi(data, explanation_size):
    combs = list(combinations(range(len(data)), 2))
    nb_combs = len(combs)
    concordance = 0
    nb_vars = len(data[0])

    for pair in combs:
        i1, i2 = pair
        for j in range(nb_vars):
            if data[i1][j] and data[i2][j]:
                concordance +=1

    vsi = round((concordance / (nb_combs * explanation_size)), 2)

    return vsi


def compute_vsi_and_jaccard_indices(data, explanation_size):
    """
        VSI and Jaccard Indices
    """

    combs = list(combinations(range(len(data)), 2))
    nb_combs = len(combs)
    concordance = 0
    nb_vars = len(data[0])
    jaccard_tmp = []
    jaccard = 0.0

    for pair in combs:
        i1, i2 = pair
        # concordance = 0
        union = intersection = 0
        for j in range(nb_vars):
            if data[i1][j] and data[i2][j]:
                intersection += 1
                concordance +=1
            if data[i1][j] or data[i2][j]:
                union += 1

        if union :
            jaccard_tmp.append(intersection/union)
        else : 
            jaccard_tmp.append(0.0)

    jaccard = round(sum(jaccard_tmp)/nb_combs, 2)
    vsi = round((concordance / (nb_combs * explanation_size)), 2)

    return vsi, jaccard


def check_stability_with_shap(data_row, background_dataset, model_to_explain, num_features, n_calls, explanation_size, nsamples, review, classif, dataset_name, model_name):

    """
    Evaluate the stability of an instance through multiple repetitions of the explanation calculation
    """
        
    dict_faithfulness = []
    dict_stability = []
    r2_score_tab = []
    precision_tab = []
    time_tab = []
  
    for r in range(n_calls):
        dict_faithfulness = []
        same_results = 0

        start_time = time.time()
        explainer = shap.KernelExplainer(model=model_to_explain, data=background_dataset.mean(0).reshape(1, background_dataset.shape[1]))

        explain_instance = data_row.reshape(1,-1)
    
        ##Calculation of SHAP values
        
        shap_values, construct_data, mask_data, used_weight = explainer.shap_values(X=explain_instance, nsamples=nsamples, l1_reg="num_features("+str(explanation_size)+")", review=review)
        end_time = time.time()

        pred = model_to_explain(explain_instance)
        expected_value = explainer.expected_value

        execution = end_time - start_time

        if not classif :
            expl = shap_values[0]

            dict_stability.append(expl)
        
            for k in range(nsamples):
                predict_g_synth = sum(np.array([expl[t] * mask_data[k][t] for t in range(num_features)])) + expected_value
                
                synth_data = construct_data[k].reshape(1,-1)
                predict_f_synth = model_to_explain(synth_data)[0]


                dict_faithfulness.append([round(used_weight[k], 3), predict_f_synth, predict_g_synth])

            dict_faithfulness = pd.DataFrame(dict_faithfulness, columns=["weight", "predict_f_synth", "predict_g_synth"])

            r2 = r2_score(dict_faithfulness["predict_f_synth"].values, dict_faithfulness["predict_g_synth"].values, sample_weight =dict_faithfulness["weight"].values)
            r2_score_tab.append(r2)
            
        else :
            nb_classes = len(pred[0])
            class_predicted = np.argmax(pred)
            expl = shap_values[class_predicted][0]
            dict_stability.append(expl)

            for k in range(nsamples):
                predict_g_synth = np.array([sum(np.array([shap_values[i][0][t] * mask_data[k][t] for t in range(num_features)])) + expected_value[i] for i in range(nb_classes)])

                synth_data = construct_data[k].reshape(1,-1)
                predict_f_synth = model_to_explain(synth_data)[0]


                same_results += int(np.where(rankdata(predict_f_synth * (-1)) == 1) == np.where(rankdata(predict_g_synth * (-1)) == 1))

                dict_faithfulness.append([round(used_weight[k], 3), predict_f_synth, predict_g_synth])

            dict_faithfulness = pd.DataFrame(dict_faithfulness, columns=["weight", "predict_f_synth", "predict_g_synth"])


            precision = same_results / nsamples

            precision_tab.append(precision)

        # explanation = list(expl)
        # explanation.append(execution)
        # explanation.insert(0, r)

    vsi, jaccard = compute_vsi_and_jaccard_indices(dict_stability, explanation_size)

    if not classif :
        return np.mean(np.array(r2_score_tab)), jaccard, vsi, execution

    else :
        return np.mean(np.array(precision_tab)), jaccard, vsi, execution 
        

def compute_explanations(models, X_train, X_test, dataset_name, random_state=42, classif=True, review=False, nbre_test=10, explanation_size=4, n_calls=20):

    random.seed(random_state)
    test_instances = random.sample(range(0, X_test.shape[0]), nbre_test)
    num_features = X_train.shape[1]

    results = dict()
    # nsamples = [10, 20, 30, 40, 50, 60, 70, 80, 90,100, 200, 500, 1000, 2000, num_samples(1, num_features), num_samples(2, num_features),num_samples(3, num_features)]

    nsamples = [10, 50, 100, num_samples(1, num_features), num_samples(2, num_features)]

    nsamples.sort()

    if review:
        version = "ST-SHAP"
    else :
        version = "SHAP"

    for item in models.items():

        jaccard_nsample = dict()
        score_nsample = dict()
        time_nsample = dict()
        vsi_nsample = dict()

        if classif :
            model = item[1].predict_proba
        else :
            model = item[1].predict


        for nsample in nsamples :
            print("\n Budget = ", nsample)
            
            tmp_score_tab = []
            tmp_jaccard_tab = []
            tmp_time_tab = []
            tmp_vsi_tab = []

            for index in test_instances :
                score, jaccard, vsi, time = check_stability_with_shap(X_test.iloc[index,:].values, X_train.values, model_to_explain=model, num_features=num_features, n_calls= n_calls,explanation_size=explanation_size, nsamples=nsample, review=review, classif=classif, dataset_name=dataset_name, model_name=item[0])
                
                tmp_score_tab.append(score)
                tmp_jaccard_tab.append(jaccard)
                tmp_vsi_tab.append(vsi)
                tmp_time_tab.append(time)

            jaccard_nsample[nsample] = round(np.mean(np.array(tmp_jaccard_tab)), 2)
            score_nsample[nsample] = round(np.mean(np.array(tmp_score_tab)), 2)
            vsi_nsample[nsample] = round(np.mean(np.array(tmp_vsi_tab)), 2)
            time_nsample[nsample] = round(max(tmp_time_tab),4)

        results[item[0]] = {"Jaccard" : jaccard_nsample, "VSI": vsi_nsample, "Score": score_nsample, "Time": time_nsample}

    # print("\nResults_{}_{} = {}".format(dataset_name, version, results))

    return results

