
import keras as K

class cf_callbacks(K.callbacks.Callback):
#on_{train, epoch, batch}_{begin, end}

    def on_train_begin(self, logs=None):
        if (logs is not None):
            keys = list(logs.keys())
            print("[cf_callbacks] Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        if (logs is not None):
            keys = list(logs.keys())
            print("[cf_callbacks] Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        if (logs is not None):            
            keys = list(logs.keys())
            print("[cf_callbacks] Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        if (logs is not None):
            keys = list(logs.keys())
            print("[cf_callbacks] End epoch {} of training; got log keys: {}".format(epoch, keys))

#     def on_test_begin(self, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] Start testing; got log keys: {}".format(keys))

#     def on_test_end(self, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] Stop testing; got log keys: {}".format(keys))

#     def on_predict_begin(self, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] Start predicting; got log keys: {}".format(keys))

#     def on_predict_end(self, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] Stop predicting; got log keys: {}".format(keys))

#     def on_train_batch_begin(self, batch, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             #print("[cf_callbacks] ...Training: start of batch {}; got log keys: {}".format(batch, keys))

#     def on_train_batch_end(self, batch, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] ...Training: end of batch {}; got log keys: {}".format(batch, keys))

#     def on_test_batch_begin(self, batch, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] ...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

#     def on_test_batch_end(self, batch, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] ...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

#     def on_predict_batch_begin(self, batch, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] ...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

#     def on_predict_batch_end(self, batch, logs=None):
#         if (logs is not None):
#             keys = list(logs.keys())
#             print("[cf_callbacks] ...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
