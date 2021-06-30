#FUNCION DONDE LE PASAS UN DATA Y PREDICE SEGUN KNEIGHBORS

def k_nearest_neighbors(data,predict,k=3):
    import numpy as np
    import warnings
    from math import sqrt
    from collections import Counter    
    if len(data)>= k:
        warnings.warn('K es menor que el numero total de elementos a votar')
    
    distances=[]
    for group in data:
        for feature in data[group]:
            #d=sqrt((feature[0]-predict[0])**2 + (feature[1]-predict[1])**2)
            #d=np.sqrt(np.sum((np.array(feauture)-np.array(predict))**2))
            d = np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([d,group])
    votes = [i[1] for i in sorted(distances)[:k]] #sorted ordena por la primera columna
    print(votes)
    
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result