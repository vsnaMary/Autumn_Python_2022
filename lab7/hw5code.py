import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

def djini(X, y):
    if len(y) == 0:
        return 0
        
    p0 = len(y[y == 0]) / len(y)
    p1 = 1 - p0
    
    return 1 - p1 ** 2 - p0 ** 2

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    
    R_m = feature_vector
    y_m = target_vector
    
    min_feature = np.min(R_m)
    max_feature = np.max(R_m)
    
    thresholds = []
    ginis = []
    threshold_best = None
    gini_best = None
    
    for t in np.linspace(min_feature, max_feature, num=100):   
        R_l = R_m[R_m < t]
        R_r = R_m[R_m >= t]
        
        y_l = y_m[R_m < t]
        y_r = y_m[R_m >= t]
        
        H_Rm = djini(R_m, y_m)
        H_Rl = djini(R_l, y_l)
        H_Rr = djini(R_r, y_r)
        
        Q = H_Rm - len(R_l) / len(R_m) * H_Rl - len(R_r) / len(R_m) * H_Rr

        if gini_best is None or Q > gini_best:
            gini_best = Q
            threshold_best = t
            
        thresholds.append(t)
        ginis.append(Q)
        
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._depth = 0

    def _fit_node(self, sub_X, sub_y, node, depth):
        if depth > self._depth:
            self._depth = depth
    
        if np.all(sub_y == sub_y[0]): # было !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / current_click
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # most_common возвращает список пар (значение, частота), например [(0, 12)]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1) # было sub_y[split]

    def _predict_node(self, x, tree):
        current_node = tree
        
        while True:
            if current_node['type'] == 'terminal':
                return current_node['class']
        
            j = current_node['feature_split']
            x_j = x[j]
            
            if x_j < current_node['threshold']:
                current_node = current_node['left_child']
            else:
                current_node = current_node['right_child']


    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)
        return self._depth

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
