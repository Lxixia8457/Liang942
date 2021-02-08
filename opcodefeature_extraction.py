from statistics import mean, median, stdev
import re
import math
import numpy as np

class ContractFeature(object):
    def __init__(self, contract_address, ponzi_or_nonponzi):
        self.contract_address = contract_address
        self.ponzi_or_nonponzi = ponzi_or_nonponzi
        self.file_name = self.contract_address + ".csv"

        # features from transactions
        self.kr = 0.0  # known rate
        self.bal = 0.0  # balance
        self.n_inv = 0 # number of investment
        self.n_pay = 0  # number of payment
        self.d_ind = 0  # difference between counts of payment and investment
        self.pr = 0.0 # the proportion of investors who received at least one payment.
        self.n_max = 0.0 # the maximum of counts of payments to participants
        self.tra_inv = 0.0  #  number of investment
        self.tra_pay = 0.0  #  number of payment
        self.n_invor = 0   #touzidizhishu
        self.n_payer = 0 #huibao dizhishu 
        self.std_pay =0.0  #fanli biaozhuncha

        # features from opcode
        self.action_freq = {}
        self.action_ratio = {}
        self.action_freq_all = {}

        # local variables
        self.investors = {}  # format: {address: first_invest_timestamp}
        self.receivers = {}  # format: {address: first_payment timestamp}
        self.payment_counts = {}  # format: {address: number_of_payment_count_to_this_address}
        self.investment_count = {}  # format: {address: number_of_investment_from_this_address}
        self.investment ={}   # format: {address: 获得投资的数量}
        self.rebate = {}

        self.get_kr_ninv_npay_pr_nmax()
        self.get_balance()

        self.get_d_ind()  # TODO
        self.get_action_frequency()  # get the frequency of each action in the opcode of a smart contract

    def get_kr_ninv_npay_pr_nmax(self):
        with open('./data/transactions/' + self.ponzi_or_nonponzi + '/' + self.file_name, 'r') as tx_file:
            counter = 0
            transaction = 0.0
            for tx in tx_file:
                if counter > 0:
                    fields = tx.split(',')
                    from_address = fields[6]
                    # print(from_address)
                    invest_timestamp = fields[2]
                    transaction += eval(fields[8])
                    # print(invest_timestamp)
                    if from_address not in self.investors:
                        self.investors[from_address] = invest_timestamp
                    else:
                        if invest_timestamp < self.investors[from_address]:
                            self.investors[from_address] = invest_timestamp

                    if from_address not in self.investment_count:
                        self.investment_count[from_address] = 1
                    else:
                        self.investment_count[from_address] += 1
                        # 
                    if from_address not in self.investment:
                        self.investment[from_address] = fields[8]
                    else:
                        self.investment[from_address] += fields[8]
                            # 

                counter += 1
            self.n_inv = counter - 1
            self.tra_inv = transaction
            self.n_invor = len(self.investors)
            # print(self.investors)
            # print('n_inv is {%d}' % self.n_inv)

        with open('./data/internal_transactions/' + self.ponzi_or_nonponzi + '/' + self.file_name, 'r') as in_tx_file:
            counter = 0
            transaction = 0.0
            for in_tx in in_tx_file:
                if counter > 0:
                    fields = in_tx.split(',')
                    to_address = fields[4]
                    pay_timestamp = fields[1]
                    transaction += eval(fields[5])

                    if to_address not in self.receivers:
                        self.receivers[to_address] = pay_timestamp
                    else:
                        if pay_timestamp < self.receivers[to_address]:
                            self.receivers[to_address] = pay_timestamp

                    if to_address not in self.payment_counts:
                        self.payment_counts[to_address] = 1
                    else:
                        self.payment_counts[to_address] += 1
                    
                    # 
                    if to_address not in self.rebate:
                        self.rebate[to_address] = eval(fields[5])
                    else:
                        self.rebate[to_address] += eval(fields[5])
                    # 
                    # print(self.rebate)

                counter += 1
            self.n_pay = counter - 1
            payment_count_list = [self.payment_counts[address] for address in self.receivers]
            self.n_max = max(payment_count_list) if len(payment_count_list) > 0 else 0
            self.tra_pay = transaction
           
            # self.std_pay = np.std(self.rebate, ddof=1)
            # print('n_max is {%d}' % self.n_max)
            # print(self.receivers)
            # print('n_pay is {%d}' % self.n_pay)
        
        for a in self.investment_count:
            if a not in self.rebate:
                self.rebate[a] = 0
        pay = list(self.rebate.values())
        # a = len(pay)
        # print(a)
        self.n_payer = len(self.receivers)
        # print(pay)
        if len(pay) == 1:
            self.std_pay = 0
        else:
            self.std_pay = stdev(pay)        
        # self.std_pay = stdev(pay) 
        
        pay_after_investment_counter = 0
        for address in self.receivers:
            if address in self.investors and self.receivers[address] > self.investors[address]:
                pay_after_investment_counter += 1
        self.kr = pay_after_investment_counter / len(self.receivers) if len(self.receivers) > 0 else 0
        # print('kr is {%f}' % self.kr)

        get_paid_investors_counter = 0
        for address in self.investors:
            if address in self.receivers:
                get_paid_investors_counter += 1
        self.pr = get_paid_investors_counter / len(self.receivers) if len(self.receivers) > 0 else 0
        # print('pr is {%f}' % self.pr)

    # get the balance of a tx from the file flaged.csv
    def get_balance(self):
        with open('./data/flaged.csv', 'r') as flaged_csv:
            for contract in flaged_csv:
                if self.contract_address == contract.strip().split(',')[0]:
                    self.bal = float(contract.strip().split(',')[3].split(' ')[0])
        # print('bal is {%f}' % self.bal)
        
    
            

    def get_action_frequency(self):
        s = 0
        with open('./data/contracts/' + self.ponzi_or_nonponzi + '/' + self.contract_address+'.txt', 'r') as code_file:
            code = code_file.read()
            text = re.sub('[\[!@#$\]]', '', code)
            lines = text.split(',')[:-5]
            for line in lines:
                s += 1
                line_action = line.strip().split(' ')[1]
                if line_action not in self.action_freq_all:
                    self.action_freq_all[line_action] = 1
                else:
                    self.action_freq_all[line_action] += 1    
        # print(self.action_freq_all)
        a = len(self.action_freq_all)
        for word in ['GASLIMIT', 'EXP', 'CALLDATALOAD', 'SLOAD', 'CALLER', 'LT', 'GAS', 'MOD', 'MSTORE']:
            # print(word)
            l = 0
            for word_all in self.action_freq_all:
                if word != word_all:
                    l += 1
            if l==a:
                self.action_freq_all[word] = 0 
                a += 1
        # print(self.action_freq_all)
        with open('./data/contracts/' + self.ponzi_or_nonponzi + '/' + self.contract_address+'.txt', 'r') as code_file:
            code = code_file.read()
            self.code_processing(code,s)
            

    def code_processing(self, text,s):
        # print(self.contract_address)
        # print(s)
        # print(text)
        text = re.sub('[\[!@#$\]]', '', text)
        # print(text)
        lines = text.split(',')[:-5]
        # print(lines)
        for line in lines:
            # print(line)
            # print(line.strip())
            line_action = line.strip().split(' ')[1]
            if line_action not in self.action_freq:
                self.action_freq[line_action] = 1
            else:
                self.action_freq[line_action] += 1

        total_num_of_actions = sum(self.action_freq.values())
        self.action_ratio['GASLIMIT'] = self.action_freq['GASLIMIT']  / total_num_of_actions if len(
            self.action_freq) > 0 and 'GASLIMIT' in self.action_freq else 0
        
        self.action_ratio['EXP'] = self.action_freq['EXP']  / total_num_of_actions if len(
            self.action_freq) > 0 and 'EXP' in self.action_freq else 0

        self.action_ratio['CALLDATALOAD'] = self.action_freq['CALLDATALOAD'] / total_num_of_actions if len(
            self.action_freq) > 0 and 'CALLDATALOAD' in self.action_freq else 0

        self.action_ratio['SLOAD'] = self.action_freq['SLOAD'] / total_num_of_actions if len(
            self.action_freq) > 0 and 'SLOAD' in self.action_freq else 0

        self.action_ratio['CALLER'] = self.action_freq['CALLER']  / total_num_of_actions if len(
            self.action_freq) > 0 and 'CALLER' in self.action_freq else 0

        self.action_ratio['LT'] = self.action_freq['LT']  / total_num_of_actions if len(
            self.action_freq) > 0 and 'LT' in self.action_freq else 0

        self.action_ratio['GAS'] = self.action_freq['GAS']  / total_num_of_actions if len(
            self.action_freq) > 0 and 'GAS' in self.action_freq else 0

        self.action_ratio['MOD'] = self.action_freq['MOD'] / total_num_of_actions if len(
            self.action_freq) > 0 and 'MOD' in self.action_freq else 0

        self.action_ratio['MSTORE'] = self.action_freq['MSTORE']  / total_num_of_actions if len(
            self.action_freq) > 0 and 'MSTORE' in self.action_freq else 0
        
           
                
        # print("********************************")

        # print(self.action_ratio['GASLIMIT'])
        # print(self.action_ratio['EXP'])
        # print(self.action_ratio['CALLDATALOAD'])
        # print(self.action_ratio['SLOAD'])
        # print(self.action_ratio['CALLER'])
        # print(self.action_ratio['LT'])
        # print(self.action_ratio['GAS'])
        # print(self.action_ratio['MOD'])
        # print(self.action_ratio['MSTORE'])

        # print("********************************")
        
        # print(self.action_freq_all['GASLIMIT'])
        # print(self.action_freq_all['EXP'])
        # print(self.action_freq_all['CALLDATALOAD']) 
        # print(self.action_freq_all['SLOAD'])
        # print(self.action_freq_all['CALLER'])
        # print(self.action_freq_all['LT'])
        # print(self.action_freq_all['GAS'])
        # print(self.action_freq_all['MOD'])
        # print(self.action_freq_all['MSTORE'])
        
        
        
        IDF_GASLIMIT = math.log((s/(self.action_freq_all['GASLIMIT']+1)),math.e) 
        IDF_EXP = math.log((s/(self.action_freq_all['EXP']+1)),math.e)        
        IDF_CALLDATALOAD = math.log((s/(self.action_freq_all['CALLDATALOAD']+1)),math.e)
        IDF_SLOAD = math.log((s/(self.action_freq_all['SLOAD']+1)),math.e) 
        IDF_CALLER = math.log((s/(self.action_freq_all['CALLER']+1)),math.e) 
        IDF_LT = math.log((s/(self.action_freq_all['LT']+1)),math.e) 
        IDF_GAS = math.log((s/(self.action_freq_all['GAS']+1)),math.e) 
        IDF_MOD = math.log((s/(self.action_freq_all['MOD']+1)),math.e) 
        IDF_MSTORE = math.log((s/(self.action_freq_all['MSTORE']+1)),math.e)
        
        # print("*****************************")
        
        # print(IDF_GASLIMIT)
        # print(IDF_EXP)
        # print(IDF_CALLDATALOAD)
        # print(IDF_SLOAD)
        # print(IDF_CALLER)
        # print(IDF_LT)
        # print(IDF_GAS)
        # print(IDF_MOD)
        # print(IDF_MSTORE)
        
        # print("*****************************")
        
        TI_GASLIMIT =self.action_ratio['GASLIMIT']*IDF_GASLIMIT
        TI_EXP =self.action_ratio['EXP']*IDF_EXP
        TI_CALLDATALOAD =self.action_ratio['CALLDATALOAD']*IDF_CALLDATALOAD
        TI_SLOAD =self.action_ratio['SLOAD']*IDF_SLOAD
        TI_CALLER =self.action_ratio['CALLER']*IDF_CALLER
        TI_LT =self.action_ratio['LT']*IDF_LT
        TI_GAS =self.action_ratio['GAS']*IDF_GAS
        TI_MOD =self.action_ratio['MOD']*IDF_MOD
        TI_MSTORE =self.action_ratio['MSTORE']*IDF_MSTORE
        
        
        # print(TI_GASLIMIT)
        # print(TI_EXP)
        # print(TI_CALLDATALOAD)
        # print(TI_SLOAD)
        # print(TI_CALLER)
        # print(TI_LT)
        # print(TI_GAS)
        # print(TI_MOD)
        # print(TI_MSTORE)
        
        # print(self.contract_address)
        
        
        self.action_ratio['GASLIMIT'] = TI_GASLIMIT      
        self.action_ratio['EXP'] = TI_EXP
        self.action_ratio['CALLDATALOAD'] = TI_CALLDATALOAD
        self.action_ratio['SLOAD'] = TI_SLOAD
        self.action_ratio['CALLER'] = TI_CALLER
        self.action_ratio['LT'] = TI_LT
        self.action_ratio['GAS'] = TI_GAS
        self.action_ratio['MOD'] = TI_MOD
        self.action_ratio['MSTORE'] = TI_MSTORE
        
        
        


    def get_d_ind(self):
        all_participants = {}
        for investor in self.investment_count:
            n_i = self.payment_counts[investor] if investor in self.payment_counts else 0
            m_i = self.investment_count[investor]
            all_participants[investor] = n_i - m_i
        for payer in self.payment_counts:
            if payer not in self.investment_count:
                n_i = self.payment_counts[payer]
                m_i = 0
                all_participants[payer] = n_i - m_i

        res = all(x == 0 for x in all_participants.values())
        if res or len(all_participants) <= 2:
            self.d_ind = 0
        else:
            v_list = list(all_participants.values())
            mean_of_v_list = mean(v_list)
            median_of_v_list = median(v_list)
            std_of_v_list = stdev(v_list)
            skewness = 3 * (mean_of_v_list - median_of_v_list) / (std_of_v_list) if std_of_v_list != 0 else 0
            self.d_ind = skewness


if __name__ == '__main__':

    # read ponzi_Contracts.csv and non_ponziContracts.csv file and find the balance of each contract by looking into
    # contractFeature = ContractFeature('0xD79B4C6791784184e2755B2fC1659eaaB0f80456', 'nonponzi')
    # print("*******************")
    contractFeature = ContractFeature('0x9d31FF892f984a83e8b342a5Ece8e8911Ed909e0', 'ponzi')
    
    


