from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

results = np.load("results_bootstrap_hlines_1.npy")
fig = plt.figure()

id_pairs = (
    ("MSRpar", "MSRpar"),
    ("SMTeuroparl", "surprise.SMTnews"),
    ("SMTeuroparl", "SMTeuroparl"),
    ("MSRpar", "surprise.OnWN"),
    ("MSRvid", "MSRvid"))

for train_id, test_id in id_pairs:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    select = results[(results["train_id"] == train_id) & 
                     (results["test_id"] == test_id) &
                     (results["min_diff"] > 0) &
                     (results["max_diff"] > 0) 
                     ]
    
    ax.plot_surface(select["min_diff"].reshape((5,5)), 
                    select["max_diff"].reshape((5,5)), 
                    select["score"].reshape((5,5)),
                    cmap=cm.coolwarm,
                    rstride=1, cstride=1,
                    )                  


    baseline = float(results[(results["train_id"] == train_id) & 
                       (results["test_id"] == test_id) &
                       (results["min_diff"] == 0) &
                       (results["max_diff"] == 0)][0][-1])
    ax.set_title("{} -> {}: baseline = {:2.2f}".format(train_id, test_id, baseline))
    ax.set_xlabel("min_diff")
    ax.set_ylabel("max_diff")
    ax.set_zlabel("r")
    
    plt.savefig("results_1_{}_{}.pdf".format(train_id, test_id))    
    plt.show()
