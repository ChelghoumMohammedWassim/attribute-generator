import warnings
import pandas as pd
import random as rnd
from Genome import Genome
from itertools import combinations
from warnings import simplefilter
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import copy

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="The least populated class in y has only .* members, which is less than n_splits=10.", category=UserWarning)


class GenGeneticAlgorithm:
    
    def __init__(self, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, rnd_state = None, evaluation_size= 0.2):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.evaluation_size = evaluation_size
        
        self.rnd_state = rnd_state
        
        self.__init_size = len(X_train.columns)
        

    
    
    def __generate_Selection(self, length):
        return rnd.choices([0, 1], k=length)
    
    
    def __generate_Population(self, size, length):
        population=[]
        selection= [1 for _ in range(self.__init_size)]+ [0 for _ in range(len(pd.DataFrame(self.X_train).columns) - self.__init_size)]
        
        population.append(
            Genome(
                selection= selection,
                fit=self.__fitness(selection, self.X_train, self.y_train)
            )
        )
        population.append(
            Genome(
                selection= selection,
                fit=self.__fitness(selection, self.X_train, self.y_train)
            )
        )
        for _ in range(size):
            selection=self.__generate_Selection(length)
            population.append(
                Genome(
                    selection = selection,
                    fit = self.__fitness(selection, self.X_train, self.y_train)))
        
        return population
    
    
    def __fitness(self, selection: list[int], X_data: pd.DataFrame, Y_data):
        sub_data = self.__get_sub_Data(X_data, selection, 1)

        if len(sub_data) > 0:
            x_train, x_test, y_train, y_test = train_test_split(sub_data, Y_data, test_size= self.evaluation_size, random_state=42)
                
            clf = DecisionTreeClassifier(random_state= self.rnd_state)

            clf.fit(x_train, y_train)
            
            predictions = clf.predict(x_test)
            precision = precision_score(predictions,y_test, average='macro')
            
            return precision*0.9 + (1-(len(x_train.columns)/ len(self.X_train.columns))) * 0.1

        return 0
    

   
   
    def __selection_pair(self, population):
        return rnd.choices(
            population=population,
            weights=[genome.fit
                     for genome in population],
            k=2
        )


    
    def __single_point_crossover(self, a: Genome, b: Genome):
        length = len(a.selection)
        if length < 2:
            return a, b
        p = rnd.randint(1, length-1)
        
        first_selection= a.selection[0:p]+b.selection[p:]
        first_child=Genome(
                        first_selection,
                        self.__fitness(first_selection, self.X_train, self.y_train))
        
        
        second_selection= b.selection[0:p]+a.selection[p:]
        second_child=Genome(
                        second_selection,
                        self.__fitness(second_selection, self.X_train, self.y_train))
        
        return  first_child, second_child
    

    
   
    def __mutation(self, genome: Genome, probability=0.15):
        
        if 0 in genome.selection:
            selection = []
            for i in genome.selection:
                if float(rnd.random()) < probability:
                    selection.append(abs(i-1))
                else:
                    selection.append(i)
            
            return Genome(selection, self.__fitness(selection, self.X_train, self.y_train))
        return Genome(genome.selection, self.__fitness(genome.selection, self.X_train, self.y_train))
   
   
   
    def __get_sub_Data(self, data: pd.DataFrame, selection , value: int):
        selected_columns = [col for col, sel in zip(data.columns, selection) if sel == value]     
        return pd.DataFrame(data[selected_columns])
    
    
    def __normalize_data(self, data: pd.DataFrame):
        min_vals = data.min()
        max_vals = data.max()
        
        normalized_data = (data - min_vals) / (max_vals - min_vals)
    
        return normalized_data
    
    
    def lasts_same(self, lst):
        if len(lst) < 5:
            return False

        last_10 = lst[-5:]

        return all(element == last_10[0] for element in last_10)
    
    
    def __generate_attribute(self, data, sets_len: int):
        operations = [self.__addition, self.__average, self.__multiplication]

        indexes = list(data.columns)

        new_data = pd.DataFrame()

        for selected_columns in combinations(indexes, sets_len):
            for operation in operations:
                selected_columns_str = [str(col) for col in selected_columns]
                new_column_name = "_".join(selected_columns_str) + "_" + operation.__name__
                new_data[new_column_name] = operation(data[list(selected_columns)])
                
        
        new_data = pd.concat([data.reset_index(drop=True), new_data.reset_index(drop=True)], axis=1)
        
        return new_data
    
    
    def __addition(self, data: pd.DataFrame):
        sums = []
    
        for index, row in data.iterrows():
            row_sum = row.sum(skipna=True)
            
            sums.append(row_sum)
        
        return pd.DataFrame(sums)


  
    def __multiplication(self, data: pd.DataFrame):
        sums = []
    
        for index, row in data.iterrows():
            row_sum = row.prod(skipna=True)
            
            sums.append(row_sum)
        
        return pd.DataFrame(sums)

    
    def __average(self, data: pd.DataFrame):
        sums = []
    
        for index, row in data.iterrows():
            row_sum = row.mean(skipna=True)
            
            sums.append(row_sum)
        
        return pd.DataFrame(sums)


    
    def run(self, population_size: int, generation_limit: int, mutation_probability=0.15):
        
        self.X_train = self.__generate_attribute(self.X_train, 2)   
        self.X_test = self.__generate_attribute(self.X_test, 2)     
                        
        population = self.__generate_Population(
            size= population_size,
            length= len(pd.DataFrame(self.X_train).columns)
            )
                
        fits=[]
        nb = 1
        
        for i in range(generation_limit):
            
            population = sorted(
                population,
                key=lambda genome: genome.fit,
                reverse=True
            )
            
                        
            next_generation = population[0:2]
            
            for _ in range(int(len(population)/2)-1):
            
                parent = self.__selection_pair(population)
                a, b = self.__single_point_crossover(parent[0], parent[1])
                
                if (self.lasts_same(fits)):
                    nb = nb+1
                    next_generation += [self.__mutation(a, 0.5), self.__mutation(b, mutation_probability)]
                
                next_generation += [self.__mutation(a, mutation_probability), self.__mutation(b, mutation_probability)]
            
            population = next_generation
                    
        population = sorted(
                population,
                key=lambda genome: genome.fit,
                reverse=True
            )
                
        test_sub_data = self.__get_sub_Data(self.X_test, population[0].selection, 1)
        
        train_sub_data = self.__get_sub_Data(self.X_train, population[0].selection, 1)
                
        return train_sub_data, test_sub_data, self.X_train, self.X_test