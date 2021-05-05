LSTM training
=============

In this tutorial, we train a recurrent neural network architecture (i.e., a stack of Bayesian LSTMs) on CDMs data, and use it for prediction purposes.

We assume that data have already been loaded (either from ``.kvn`` format, from pandas ``DataFrame`` object, or from the Kelvins challenge dataset: see the relevant tutorials) and stored into ``events``.
We can then first define the features that have to be taken into account during training: this is a list of feature names. In this case, we can take all the features present on the uploaded data, provided that they have numeric content:

.. code-block:: python
   
   nn_features=events.common_features(only_numeric=True)

We can then split the data into test (here defined as 5% of the total number of events) and training & validation set:

.. code-block:: python

   len_test_set=int(0.05*len(events))
   events_test=events[-len_test_set:]
   events_train_and_val=events[:-len_test_set]

Finally, we create the LSTM predictor, by defining the LSTM hyperparameters as we wish:

.. code-block:: python

   from kessler.nn import LSTMPredictor
   model = LSTMPredictor(
               lstm_size=256, #number of hidden units per LSTM layer
               lstm_dept=2,   #number of stacked LSTM layers
               dropout=0.2,   #dropout probability
               features=nn_features) #the list of feature names to use in the LSTM

Then we start the training process:

.. code-block:: python

   model.learn(events_train_and_val,
               epochs=10, #number of epochs
               lr=1e-3, #learning rate (can decrease if training diverges)
               batch_size=16, #minibatch size (can decrease if there are memory issues)
               device='cpu', #can be 'cuda' if there is a GPU available
               valid_proportion=0.15, #proportion of data used as validation set
               num_workers=4, #number of multithreaded dataloader workers (usually 4 is good for performances, but if there are issues, try 1)
               event_samples_for_stats=1000) #number of events to use to compute NN normalization factors

Finally, we save the model to a file after training, and we plot the validation and training loss and save the image to a file:

.. code-block:: python

   model.save(file_name='LSTM_20epochs_lr1e-4_batchsize16')
   model.plot_loss(file_name='plot_loss.pdf')

We now test the prediction. We take a single event, we remove the last CDM and try to predict it:

.. code-block:: python

   event=events_test[3]
   event_len=len(event)
   event_beginning=event[0:event_len-1]
   event_evolution=model.predict_event(event_beginning, num_samples=100, max_length=14)
   #we plot the prediction in red:
   axs=event_evolution.plot_features(['RELATIVE_SPEED', 'MISS_DISTANCE'], return_axs=True, linewidth=0.1, color='red', alpha=0.33, label='Prediction')
   #and the ground truth value in blue:
   event.plot_features(['RELATIVE_SPEED', 'MISS_DISTANCE'], axs=axs, label='Real', legend=True)

