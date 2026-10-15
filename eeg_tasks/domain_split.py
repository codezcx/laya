"""Whole-source MI validation, keeping original source test entirely excluded."""
def domain_indices(rows,heldout):
    if heldout not in {'Cho2017','Dreyer2023','Lee2019_MI'}:raise ValueError(heldout)
    train=[];cross=[];within=[]
    for i,r in enumerate(rows):
        if r['split'] not in ('train','validation'):raise ValueError('Only source development rows allowed')
        if r['task']==0:
            if r['split']=='train':train.append(i)
            else:cross.append(i);within.append(i)
        elif r['task']==1:
            if r['dataset']==heldout:cross.append(i)
            elif r['split']=='train':train.append(i)
            else:within.append(i)
        else:raise ValueError('Inactive task')
    assert train and cross and within and not set(train)&(set(cross)|set(within))
    assert all(rows[i]['dataset']!=heldout for i in train)
    assert not {(rows[i]['dataset'],rows[i]['subject']) for i in train}&{(rows[i]['dataset'],rows[i]['subject']) for i in cross+within}
    return train,cross,within
