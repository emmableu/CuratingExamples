from scipy.stats import chi2_contingency
import numpy as np
from scipy import stats


class Test:
    def __init__(self, data):
        '''
        input data format:
        pd.DataFrame
        index: some default pd index
        columns: ['pid', 'pattern occurance' (aka 'occurance'), 'label']
        example: [[343434342, 3, yes],
                    [dfdfdvc, 0, no]]
        '''
        self.data = data
        # print("Train data: ", data)
        self.yes = self.data[self.data.code == 'yes']
        self.no = self.data[self.data.code == 'no']
        self.yes_occurance_list = self.yes['occurance'].to_list()
        self.no_occurance_list = self.no['occurance'].to_list()
        self.yes_total = len(self.yes_occurance_list)
        self.no_total = len(self.no_occurance_list)
        self.pattern_exist_yes = len([i for i in self.yes_occurance_list if i > 0])
        self.pattern_non_exist_yes = len([i for i in self.yes_occurance_list if i == 0])
        self.pattern_exist_no = len([i for i in self.no_occurance_list if i > 0])
        self.pattern_non_exist_no = len([i for i in self.no_occurance_list if i == 0])

    def get_freq(self, label):



        if label == "yes":
            return self.pattern_exist_yes/self.yes_total
        else:
            return self.pattern_exist_no/self.no_total


    def freq_compare_test(self):
        '''
        only significant if left >= 1.3 right
        :return: true means significant, False means unsignificant
        '''

        yes_freq, no_freq = self.get_freq("yes"), self.get_freq("no")
        if yes_freq >= 1.3* no_freq:
            return True
        elif yes_freq < no_freq:
            return "discard"
        else:
            return False

    def chi_square_test(self):
        try:
            obs = np.array([[self.pattern_exist_yes, self.pattern_non_exist_yes], [self.pattern_exist_no, self.pattern_non_exist_no]])
            chi2, p, dof, ex = chi2_contingency(obs, correction=False)
            if p <= 0.3:
                return True
            else:
                return False
        except:
            return False

    def kruskal_wallis_test(self):
        try:
            p = stats.kruskal(self.yes_occurance_list, self.no_occurance_list).pvalue
            if p <= 0.2:
                return True
            else:
                return False
        except:
            return False





