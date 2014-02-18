'''
        2 * MaP * MaR
MaF =  ---------------
          MaP + MaR

P = tp / (tp + fp)
R = tp / (tp + fn)
MaP = average([P for every category])
MaR = average([R for every category])

'''
from collections import defaultdict

class CategoryPNCounter(defaultdict):
    def __init__(self):
        super(CategoryPNCounter, self).__init__(lambda : defaultdict(int))

    def fill_pns(self, v_labels, t_labels):
        ''' fill in pn's '''
        v_labels_set, t_labels_set = set(v_labels), set(t_labels)
        tps = t_labels_set & v_labels_set
        fps = v_labels_set - t_labels_set
        fns = t_labels_set - v_labels_set
        for label in tps:
            self.__getitem__(label)['tp'] += 1
        for label in fps:
            self.__getitem__(label)['fp'] += 1
        for label in fns:
            self.__getitem__(label)['fn'] += 1

    def calculate_cat_pr(self):
        ''' For each category, calculate p & r '''
        for cat, ddict in self.iteritems():
            tp = ddict.__getitem__('tp')
            fp = ddict.__getitem__('fp')
            fn = ddict.__getitem__('fn')
            p = tp / (tp + fp) if tp+fp != 0 else 0
            r = tp / (tp + fn) if tp+fn != 0 else 0
            self.__getitem__(cat)['p'] = p
            self.__getitem__(cat)['r'] = r

        self.MaP = sum(ddict.__getitem__('p') for ddict in self.itervalues()) \
                       / len(self)
        self.MaR = sum(ddict.__getitem__('r') for ddict in self.itervalues()) \
                       / len(self)
        return self.MaP, self.MaR

    def calculate_MaF(self):
        ''' Calculate Macro-F1 score.
            `calculate_cat_pr` must be run first! '''
        if self.MaP == 0 or self.MaR == 0:
            self.MaF = 0.0
        else:
            self.MaF = 2 * self.MaP * self.MaR / (self.MaP + self.MaR)
        return self.MaF



