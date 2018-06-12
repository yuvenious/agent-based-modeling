import math
import random
import itertools

import numpy as np
import pandas as pd

class Model():
    
    def __init__(self, n_mun, target=0, percept=0, knowledge=0, frac_r_max=0.3, frac_p_max=0.8,
                 budget_camp_prop=0.5, infra=0, ambitious=0.5, print_year=True):
        """Model class has following input parameters
        1. n_mun: number of municipality
        2. n_com: number of company
        3. dist: population distribution (single, couple, family, retired)"""
        self.n_mun = n_mun
        self.n_com = n_mun * 5 # for every municipality, 5 company will be matched
        self.dist = [0.1, 0.4, 0.3, 0.2]
        """Model class has following attributes:
        - ticks: how much time has passed
        - mun_dict: dictionary of municipality objects
        - com_dict: dictionary of company objects
        - requests: requests posted by Municipality
        - offers: bid offers from Company"""
        self.ticks = 0
        self.mun_dict = dict()
        self.com_dict = dict()
        self.requests = []
        self.offers = []
        
        # for the sake of logging
        self.log = []
        self.name = "model"
        
        """MODEL PARAMETERS"""
        self.target = target
        self.knowledge = knowledge
        self.percept = percept
        
        self.budget_camp_prop = budget_camp_prop
        
        self.frac_r_max = frac_r_max
        self.frac_p_max = frac_p_max
        
        self.infra = infra
        self.ambitious = ambitious
        
        self.print_year = print_year
        
    def populate(self):
        self.log.append([self.ticks, self.name, "Populate the Model"])
        
        """Populate Municipalities
        two parameters (n_pop, dist) determine how each municipality look like        
        1. n_pop: number of population
        2. dist: distribution ["single", "couple", "family", "retired"]"""
        for i in range(self.n_mun):
            """Population is between 75~500k,
            given 1) annual waste: 3k~20kTon and 2) (avg) 40kg per year per household"""
            mul = 1
            n_pop = random.randint(75*mul, 500*mul)
            
            # facility 50% to 80% by default
            infra = self.infra
            infra = random.uniform(0.5+infra, 0.8+infra)
            # target to be met
            target = self.target
            target = random.uniform(0.20+target, 0.24+target)
            # fine threshold
            fine_thresh = random.uniform(0.005, 0.01)
            # ambitious
            ambitious = self.ambitious
            ambitious = random.random() < ambitious
            
            m = self.Municipality(i, n_pop, self.dist, infra, target, fine_thresh, ambitious, self)
            # store into the dictionary
            self.mun_dict[i] = m           
        
        """Populate Companies"""
        for i in range(self.n_com):
            c = self.Company(i, self)
            # store into the dictionary
            self.com_dict[i] = c
            
    def update(self):
        """Before proceed, Contract has to be made in following orders:
        1. Municipality: Request -> 2. Company: Offer (Bid) -> 3. Municipality: Select winner"""
        
        # 1. Municipality Requests
        self.contract_mun()
        if len(self.requests) != 0:
            self.requests = np.array(self.requests)
            self.log.append([self.ticks, self.name, "Requests Posted on the Wall"])
        else:
            self.log.append([self.ticks, self.name, "No Request"])
            
        # 2. Company Offers
        self.contract_com()
        if len(self.offers) != 0:
            self.offers = pd.DataFrame(self.offers)
            self.log.append([self.ticks, self.name, "Offers Posted on the Wall"])
        else:
            self.log.append([self.ticks, self.name, "No Offers"])
            
        # 3. Municipality Selects and Closes bid
        self.select_mun()
        # Requests and Offers are initialized after bid closed
        if len(self.requests) != 0:
            self.requests = []
            self.offers = []
        
        """Update the model by proceeding time tick
        Each municipality step forward (and Households inside)"""
        self.update_mun()
        self.update_com()
        
        self.log.append([self.ticks, self.name, "Ticks %s Ends"%self.ticks])
        self.ticks += 1
        
        # Print the yearly time progress
        if self.print_year:
            if self.ticks%12 == 0:
                print("%s year(s) passed"%(int(self.ticks/12)))

    def contract_mun(self):
        # Municipalities REQUEST
        for i, mun in self.mun_dict.items():
            mun._check_contract()
            
    def contract_com(self):
        # Companies OFFER (bid)
        for i, com in self.com_dict.items():
            com._check_request()
    
    def select_mun(self):
        # Municipalities SELECT bidder
        for i, mun in self.mun_dict.items():
            if mun.bid_on:
                mun._select()

    def update_mun(self):
        # Municipalities step forward
        for i, mun in self.mun_dict.items():
            mun._update()
            
    def update_com(self):
        # Companies step forward
        for i, com in self.com_dict.items():
            com._update()
            
    """COMPANY"""            
    class Company():
        
        def __init__(self, unique_id, model):
            self.unique_id = unique_id
            self.name = " ".join(["Company", str(self.unique_id)])
            self.model = model
            
            self.contract_made = []
            self.have_contract = False
            self.contract_history = []
            
            self.profit = dict()
            self.burned = dict()
            
            # technology level is randomly assigned
            self.tech = random.random()
            if self.tech < 0.3:
                self.level = "low"
            elif self.tech < 0.7:
                self.level = "med"
            else:
                self.level = "high"
            
        def _check_request(self):
            """Check whether there is a request"""
            if len(self.model.requests) != 0:
                self.model.log.append([self.model.ticks, self.name, "Read Requests"])
                self._offer()
        
        def _offer(self):
            self.model.log.append([self.model.ticks, self.name, "Make Offer"])
            """Company makes an offer"""
            requests = self.model.requests
            request = requests[requests[:,0] == self.unique_id//5][0]
            
            # Company Reads the requirments of Municipalities (volume and percent)
            mun_id = request[0]
            X = request[1]
            x = request[2]

            """Based on the volume and percent (posted on the requests), Company creates an offer"""
            X = random.gauss(X, sigma = X / 20)
            x = random.gauss(x, sigma = 0.01)
            bidprice = int(X * x)
            
            # Company suggests fine 0.5~1% of the contract price as a contingency plan
            fine = bidprice * random.uniform(0.005, 0.01)
            
            # Minimum Requirement: waste must contain at least same amount of the plastics, otherwise: fine
            min_plastic = self.model.mun_dict[mun_id]._plastic
            
            # Create the bid package and post on the market (model)
            offer_key = ["mun_id", "vol", "perc", "min_plastic", "bidprice", "fine", "com_id", "time"]
            offer_list = [self.unique_id//5, X, x, min_plastic, bidprice, fine, self.unique_id, self.model.ticks]
            self.model.offers.append(dict(zip(offer_key, offer_list)))
            
        def _receive_contract(self, contract):
            self.contract_made = contract
            mun_id = self.contract_made["mun_id"]
            self.client = self.model.mun_dict[mun_id]
            self.model.log.append([self.model.ticks, self.name, "Receive contract from %s"%(self.client.name)])
            self.have_contract = True # Company works only if they have a contract
            self.contract_history.append(self.contract_made)
            
        def _update(self):
            if self.have_contract:
                self.model.log.append([self.model.ticks, self.name, "Have contract → Fine and Recover"])
                self._fine()
                self._recover()
                            
            else:
                self.model.log.append([self.model.ticks, self.name, "No contract → Pass"])
                # no profit, no wastes to burn
                self.profit[self.model.ticks] = 0
                self.burned[self.model.ticks] = 0
            
        def _fine(self):
            # Specify Municipality in Contract
#             mun_id = self.contract_made["mun_id"]
#             mun = self.model.mun_dict[mun_id]
            
            mun = self.client
            
            # Check Wastes from Municipality
            waste = mun._waste
            plastic = mun._plastic
            recycle = mun._recycle
            
            # Compare with Contract
            self.model.log.append([self.model.ticks, self.name, "Compare Waste vs. Contract: %s vs. %s"%(plastic, self.contract_made["min_plastic"])])
            claim_fine = self.contract_made["min_plastic"] > plastic
            
            # Claim Fine
            if claim_fine:
                fine = self.contract_made["fine"]
                self.model.log.append([self.model.ticks, self.name, "Claim Fine %s to %s"%(fine, mun.name)])
                mun.fine = True
            # No Fine
            else:
                fine = 0
                self.model.log.append([self.model.ticks, self.name, "No Fine to %s"%(mun.name)])
                mun.fine = False
            mun.fine_history[self.model.ticks] = self.unique_id, fine
            
        def _recover(self):
            self.model.log.append([self.model.ticks, self.name, "Recover Wastes of %s"%(self.client.name)])
            
            # 10% margin of recyclable plastics, based on their technology
            profit = 0.1 * self.client._recycle * self.tech
            self.profit[self.model.ticks] = profit
            
            # burn the rest
            burned = self.client._plastic - self.client._recycle
            self.burned[self.model.ticks] = burned
            
    """MUNICIPALITY"""                
    class Municipality():
        
        def __init__(self, unique_id, n_pop, dist, infra, target, fine_thresh, ambitious, model):
            """
            Municipality
            1. ID
            2. n_pop: Population
            3. dist: Distribution
            4. infra: percentage of infrastructure
            5. budget
            4. pop: Population Distribution (Dictionary object as a result of 2, 3)
            5. household_dict: Residents
            6. Model object (to easily access to the upper class variable)
            
            7. Waste History (Waste, Plastic, Recycle)
            
            (After update)
            8. (Sum) Waste
            9. (Sum) Plastic Waste
            10. (Sum) recyle
            
            11. target (to be met)
            12. provider (waste service, company)
            13. acceptable fine (threshold)
            
            14. contract history
            15. fine history
            """
            self.unique_id = unique_id
            self.name = " ".join(["Municipality", str(self.unique_id)])
            self.n_pop = n_pop
            self.dist = dist
            
            self.infra = infra
            # account book (initially blank)
            self.budget = np.zeros((240, 3))
            
            self.pop = np.array(dist) * n_pop
            self.household_dict = dict()
            self.model = model

            self.waste_history = []
            
            self.target = target
            self.fine_thresh = fine_thresh
            
            self.bid_on = False
            self.contract_history = []
            self.provider = 0
            self.fine = False
            self.fine_cumsum = 0
            self.fine_history = dict()
            
            # The more ambitious, the more budget use, the larger effect (240 months * 0.005 effects ≒ 1.2)
            self.ambitious = ambitious
            if self.ambitious:
                self.aimed_months = random.randint(60, 120) # 5~10 years target
                self.camp_effect = 0.005 * 240/self.aimed_months
            else:
                self.aimed_months = random.randint(120, 180) # 10~15 years target
                self.camp_effect = 0.005 * 240/self.aimed_months

            self.target_met = False
            
            for i in range(self.n_pop):
                """When the municipality initializes,
                it automatically populate households inside"""
                # each house hold takes: ID, Municipal ID and Municipal object
                h = self.Household(i, self.unique_id, self)
                # Automatically Populate by the parameter: distribution (in order to decide type)
                h._populate(self.dist)
                # store inside the dictionary
                self.household_dict[i] = h
            
        def _check_contract(self):
            if self.model.ticks%36 == 1: 
                self.bid_on = True
                self.model.log.append([self.model.ticks, self.name, "Need Contract"])
                self._request()
                
        def _request(self):
            """Municipality request offer"""
            if self.provider != 0:
                self._term_contracts()
            elif self.provider == 0: # the first moment
                self._set_budget()
            
            # We have X volumn, x% should be recycled
            X = self._waste
            x = self.target
            # Change the environment (Model)
            # Posting a Bid
            self.model.log.append([self.model.ticks, self.name, "Post Bid X:%s, x:%s"%(X, x)])
            self.model.requests.append([self.unique_id, X,x])
        
        def _set_budget(self):
            # Contract budget: Total (240 months estimated) contract price (10% more)
            budget_contract = 1.1 * self._waste * self.target / 36 * 240
            
            # total budget for campaign is 50% of budget for contract
            budget_camp_prop = self.model.budget_camp_prop
            budget_camp = budget_contract * budget_camp_prop
            
            # monthly budget for campaign varies depending on their aim
            self.montly_camp_budget = budget_camp / self.aimed_months
            
            # budget for fine as a contingency plan, 1% of budget for contract
            budget_fine = budget_contract * 0.01 * 36
            
            self.budget[self.model.ticks-1, :] = np.array([budget_contract, budget_camp, budget_fine])
            self.model.log.append([self.model.ticks, self.name, "Set Total Budget: %s"%(self.budget[0])])
        
        def _term_contracts(self):
            """Municipality Terminates Contract Every 3 years"""
            self.model.log.append([self.model.ticks, self.name, "terminate former contract with %s"%self.provider.name])
            self.provider.contract_made = 0
            self.provider.client = None
            self.provider.have_contract = False
            
        def _select(self):
            self.model.log.append([self.model.ticks, self.name, "Evaluate Offers"])
            """Municipality evaluates offers from Companies and filters out"""
            # 1.filter mun_id
            offers = self.model.offers[self.model.offers["mun_id"] == self.unique_id]
                
            # 2.filter volume
            # if all bids doesn't match, then there is no way. just accept.
            vol_filter = offers["vol"] >= self._waste
            if len(vol_filter.nonzero()[0]) != 0:
                offers = offers[vol_filter]
                
            # 3. filter percentage
            perc_filter = offers["perc"] >= self.target
            if len(perc_filter.nonzero()[0]) != 0:
                offers = offers[perc_filter]
                
            # 4. filter fine
            fine_filter = offers["fine"] <= offers["bidprice"] * self.fine_thresh
            
            # if all bids doesn't match, then there is no way. just accept
            if len(fine_filter.nonzero()[0]) != 0:
                offers = offers[fine_filter]

            # 5. finally select the cheapest bid
            min_ = (offers["bidprice"] + offers["fine"]).idxmin()
            offers = offers.loc[min_]
            self.contract_made = dict(offers)
                
            # 6. Announce to the company who wins
            winner = offers["com_id"]
            contract_price = offers["bidprice"]
            self.model.log.append([self.model.ticks, self.name, "Select winner: Company %s, Bidprice: %s"%(winner, contract_price)])
            self.provider = self.model.com_dict[winner]
            self.provider._receive_contract(self.contract_made)
            
            # 7. Archive to its own storage
            self.contract_history.append(self.contract_made)
                        
            # 8. Bid process is over
            self.bid_on = False
                
        def _update(self):
            """When the municipality is updated,
            It forces every household to be updated"""
            for i, household in self.household_dict.items():
                household._update()
            
            # Sum up wastes from all households
            self._waste_calculate()
            
            # Montly Payment (to Company in contract)
            if self.provider != 0:
                self._payment()
            
            # Calculate fine
            if self.fine:
                self.model.log.append([self.model.ticks, self.name, "Got fine"])
                self._fine_calculate()
            # record the fine
            else:
                if self.model.ticks > 2:
                    self.budget[self.model.ticks-1, 2] = self.budget[self.model.ticks-2, 2]
                
            # Campaign about More/Better Recycling
            if self.model.ticks > 0:
                if self.budget[self.model.ticks-1, 1] > 0:
                    # Municiapality spends budget, since the target is not met yet
                    if not self.target_met:
                        self.model.log.append([self.model.ticks, self.name, "Do Campaign"])
                        self._campaign_know()
                        self._campaign_perc()
                        # Municipality has a certain months to accomplish the target
                        # The more ambitious, the quicker consume the budget
                        self.budget[self.model.ticks, 1] = self.budget[self.model.ticks-1, 1] - self.montly_camp_budget
                    # Municiapality spends no budget, since the target is met
                    else:
                        self.budget[self.model.ticks, 1] = self.budget[self.model.ticks-1, 1]
                else: # Municipality cannot do campains, as there is no budget
                    self.model.log.append([self.model.ticks, self.name, "No budget for Campaign"]) 
                    
        def _payment(self):
            montly_payment = self.contract_made["bidprice"] / 36
            self.budget[self.model.ticks, 0] = self.budget[self.model.ticks-1, 0] - montly_payment

        def _waste_calculate(self):
            """Collect all Wastes data from Households"""
            self._waste = 0
            self._plastic = 0
            self._recycle = 0
            
            # differentiate between house/facility
            self._waste_house = 0
            self._waste_facility = 0
            
            self._plastic_house = 0
            self._plastic_facility = 0
            
            self._recycle_house = 0
            self._recycle_facility = 0

            self.model.log.append([self.model.ticks, self.name, "Collect the amount of wastes produced"])
            for i, household in self.household_dict.items():
                if household.access:
                    self._waste_facility += household.waste
                    self._plastic_facility += household.plastic
                    self._recycle_facility += household.recycle
                else:
                    self._waste_house += household.waste
                    self._plastic_house += household.plastic
                    self._recycle_house += household.recycle
            
            # if there is no infrastructure, only 70% are collected
            self._waste = self._waste_house*0.7 + self._waste_facility
            self._plastic = self._plastic_house*0.7 + self._plastic_facility
            self._recycle = self._recycle_house*0.7 + self._recycle_facility
            self.waste_history.append([self._waste, self._plastic, self._recycle])
            
            self.plastic_prop = self._plastic / self._waste
            self.recycled_prop = self._recycle / self._waste
            self.model.log.append([self.model.ticks, self.name, "Calculate Proportion of Recycled Wastes: %s"%(self.recycled_prop)])
            
            # Check Whether the Target has been met
            if self.recycled_prop >= self.target:
                self.target_met = True
                self.model.log.append([self.model.ticks, self.name, "Target Met!: Target %s vs. Record %s"%(self.target, self.recycled_prop)])
            else:
                self.model.log.append([self.model.ticks, self.name, "Target Not Met: Target %s vs. Record %s"%(self.target, self.recycled_prop)])

        def _fine_calculate(self):
            # fine received at last time tick
            fine = self.fine_history[self.model.ticks - 1][1]
            self.fine_cumsum += fine
            self.budget[self.model.ticks-1, 2] = self.budget[self.model.ticks-2, 2] - fine
                        
        def _campaign_perc(self):
            """more recycling: perception"""
            self.model.log.append([self.model.ticks, self.name, "Perception Campaign"])
            before_camp = 0
            after_camp = 0
            
            for i, household in self.household_dict.items():
                before_camp += household.percept
                household._increase_percept(self.camp_effect)
                after_camp += household.percept

            before_camp = before_camp / self.n_pop
            after_camp = after_camp / self.n_pop

            self.model.log.append([self.model.ticks, self.name, "Average Perception: Before: %s, After:%s"%(before_camp, after_camp)])
            
        def _campaign_know(self):
            """better recycling: knowledge"""
            self.model.log.append([self.model.ticks, self.name, "Knowledge Campaign"])
            before_camp = 0
            after_camp = 0
            
            for i, household in self.household_dict.items():
                before_camp += household.knowledge
                household._increase_knowledge(self.camp_effect)
                after_camp += household.knowledge
            
            before_camp = before_camp / self.n_pop
            after_camp = after_camp / self.n_pop
                
            self.model.log.append([self.model.ticks, self.name, "Average Knowledge: Before: %s, After:%s"%(before_camp, after_camp)])
                                
        class Household():
            """Households are defined inside the Municipality Class"""            
            def __init__(self, unique_id, mun_id, mun_obj):
                """Household has following attributes:
                1. ID
                2. Municipal ID (where they belong to)
                3. Municipal object (for easy access to the upper class variables)
                
                (After Populate:)
                4. type ("single, couple, family, old")
                5. weights applied to the volumn of wastes
                6. access to facility
                7. perception and knowledge
                8. fraction of plastic in the waste and recyclable plastic in the plastic waste
                
                (After Update:)
                9. Wastes
                10. Plastic Wastes
                11. Recyclable wastes"""
                self.unique_id = unique_id
                self.mun_id = mun_id
                self.mun_obj = mun_obj
                
                # Limits
#                 self.frac_p_max = 0.3
#                 self.frac_r_max = 0.8

                self.frac_p_max = self.mun_obj.model.frac_p_max
                self.frac_r_max = self.mun_obj.model.frac_r_max
                
            def _populate(self, dist):
                # assign type
                r = random.random()
                id_ = np.where(np.array(dist).cumsum() > r)[0].min()
                self.type = ["0single", "1couple", "2family", "3retired"][id_]
                # each household has different weight according to its type
                self.weight = [0.95, 1.05, 1.10, 0.90][id_]
                # assign access
                self.access = random.random() < self.mun_obj.infra
                # assign percept [0,1]
                percept = self.mun_obj.model.percept
                self.percept = random.uniform(0+percept, 0.3+percept)
                self.frac_p = min(self.percept, self.frac_p_max)
                # assign knowledge [0,1]
                knowledge = self.mun_obj.model.knowledge
                self.knowledge = random.uniform(0+knowledge, 0.3+knowledge)
                self.frac_r = min(self.percept, self.frac_r_max)
                
            def _update(self):
                self.ticks = self.mun_obj.model.ticks
                self._produce()

            def _produce(self):
                x = self.ticks
                self.waste = self.weight * (40 - 0.04 * x - math.exp(-0.01*x) * math.sin(0.3*x))/12
                self.plastic = self.waste * self.frac_p
                self.recycle = self.plastic * self.frac_r
                
            def _increase_percept(self, effect):
                if (self.percept < 1.0) & (random.choice([True, False])): # 50% chance to increase
                    # increase by the effect, but cannot exceed maximum
                    self.percept = min(self.percept + effect, 1.0)
                    self.frac_p = min(self.percept, self.frac_p_max)

            def _increase_knowledge(self, effect):
                if (self.knowledge < 1.0) & (random.choice([True, False])): # 50% chance to increase
                    # increase by the effect, but cannot exceed maximum
                    self.knowledge = min(self.knowledge + effect, 1.0)
                    self.frac_r = min(self.knowledge, self.frac_r_max)

# as a Function
def recycle_model(n_mun=2, target=0, percept=0, knowledge=0, frac_r_max=0.3, frac_p_max=0.8, budget_camp_prop=0.5, infra=0, ambitious=0.5):
    model = Model(n_mun, target=target, percept=percept, knowledge=knowledge,
                  frac_r_max=frac_r_max, frac_p_max=frac_p_max,
                  budget_camp_prop=budget_camp_prop, infra=infra, ambitious=ambitious, print_year=False)
    model.populate()
    for i in range(240):
        model.update()
        
    # 4 outcomes
    plastic_prop = []
    recycle_prop = []
    target_met = []
    camp_bud_left = []

    for i, mun in model.mun_dict.items():
        # plastic prop
        plastic_prop.append(mun.plastic_prop)
        # recycle prop
        recycle_prop.append(mun.recycled_prop)
        # whether municipality's target has been met
        target_met.append(mun.target_met)
        # budgets left
        camp_bud_left.append((mun.budget[-2] / mun.budget[0])[1])


    outcomes = pd.DataFrame(np.array([plastic_prop,recycle_prop,target_met,camp_bud_left]),
                            index = ["plastic_prop", "recycle_prop", "target_met", "camp_bud_left"]).T.mean().values
    plastic_prop = outcomes[0]
    recycle_prop = outcomes[1]
    target_met = outcomes[2]
    camp_bud_left = outcomes[3]

    return plastic_prop,recycle_prop,target_met,camp_bud_left