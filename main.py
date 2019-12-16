import pandas as pd
from network import greedy

# Load Data Tables
train = pd.read_csv(
    'hw2_train.data', 
    header=None, 
    sep="\t"
)

test = pd.read_csv(
    'hw2_test.data', 
    header=None, 
    sep="\t"
)

# Run Simulation 5 times, print best network from each trial
# and tabulate results. At each trial, the network is re
# initialized to complete independence

r = False
for i in range(1,6):
    g = greedy(train)
    fail = g.train(iterations = 300, maxmiss=25)
    if not r:
        t_res = pd.DataFrame(g.scores).assign(Trial = i)
        result = g.test_accuracy(test, target=6).assign(Trial = i)
        r = True
    else:
        result = result.append(g.test_accuracy(test, target=6).assign(Trial = i))
        t_res = t_res.append(pd.DataFrame(g.scores).assign(Trial = i))

        # Print out The top network from each trial
    print("Most Recent Best Network for Trial {0}".format(i))
    g.net.print_nodes()
    print(
    	"Best Score observed in Trial {0}: {1}".format(
    	i, max(t_res['Score'].loc[t_res['Trial'] == i])
    	)
    )
    print()        
        
# Print Summary Results
print()
print("Percentage of Predictions by Trial")
result['Percent'] = result[['Trial','Correct']].groupby(
    'Trial').transform(lambda x: 100 * x.sum() / x.count())
print(result[['Trial', 'Percent']].groupby('Trial').first())
    

# Save results to a file
t_res.drop('Network', axis = 1).to_csv(
    "Train_Results.txt", sep="\t", index=False)

result.drop('Percent', axis = 1).to_csv(
    "Test_Results.txt", sep="\t", index=False)



    




