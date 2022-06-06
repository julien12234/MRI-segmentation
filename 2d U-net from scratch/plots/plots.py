import numpy as np
import matplotlib.pyplot as plt

def plot_train_val(m_train, m_val, metric, period=4):

    """Plot the evolution of the metric evaluated on the training  and validation set during the trainining
    Args:
        m_train: history of the metric evaluated on the train 
        m_val: history of the metric evaluated on the val 
        period: number of epochs between 2 valutation of the train
        metric: metric used (e.g. Loss, Dice)
    Returns:
        plot
    """

    # remove first element of the arrays
    m_train = m_train[2:]
    m_val = m_val[2:]
    
    plt.title('Evolution of the '+ metric+ ' with respect to the number of epochs',fontsize=14)
    
    plt.plot(np.array(range(0,len(m_train)))*period, m_train, color='blue', marker='o', ls=':', label=metric+' train')
    plt.plot(np.array(range(0,len(m_val)))*period, m_val, color='red', marker='o', ls=':', label=metric+' val')

    plt.xlabel('Number of Epochs')
    plt.ylabel(metric)
    plt.legend(loc = 'lower right')
    plt.savefig('evol_'+metric)
