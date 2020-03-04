class Test:
    def __init__(self, data):
        '''
        input data format:
        pd.DataFrame
        index: some default pd index
        columns: ['pid', 'pattern occurance', 'label']
        example: [[343434342, 3, yes],
                    [dfdfdvc, 0, no]]
        '''
        self.data = data

    def get_yes(self):
        yes = data[data.label == 'yes']
        return yes

    def get_no(self):
        no = data[data.label == 'no']
        return no

    def freq_compare_test(self):
        '''
        only significant if left >= 1.3 right
        :return: true means significant, False means unsignificant
        '''





