from model_type.bases.classification.baseClassifier import baseClassifier

base_clf = baseClassifier()


def single_classifications(image_path, det_weight, clf_weight, 
                           det_conf_thres, clf_conf_thres, device):
    base_clf.load_models(det_weight=det_weight, clf_weight=clf_weight, device = device)

    clf_result = base_clf.single_classification(img_path=image_path, det_conf_thres=det_conf_thres, 
                                   clf_conf_thres=clf_conf_thres)
    return clf_result
    

def batch_classifications(data_path, extension, det_weight, clf_weight, 
                          det_thres, clf_thres, device):
    base_clf.load_models(det_weight=det_weight, clf_weight=clf_weight, device = device)

    base_clf.batch_classification(data_path=data_path, det_thres = det_thres, 
                                  clf_thres=clf_thres, extension=extension, device=device)
    