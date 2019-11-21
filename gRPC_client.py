# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:40:53 2019

@author: minmhan
"""

import numpy
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
from absl import flags

import pickle
from bert import tokenization
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "ip_address", None,
    "ip address of tensorflow serving. ")
flags.DEFINE_integer(
    "port_no", 8500,
    "port no of tensorflow serving.")
flags.DEFINE_string(
    "model_name", None,
    "model name of tensorflow serving")
flags.DEFINE_string(
    "sentence", None,
    "input sentence")

def get_feature_vector(sentence, max_seq_length=128):
    text = tokenization.convert_to_unicode(sentence)
    vocab = tokenization.load_vocab("cased_L-12_H-768_A-12/vocab.txt")
    wpt = tokenization.WordpieceTokenizer(vocab)
    tokens = wpt.tokenize(text)
    
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
    
    tokens.insert(0,"[CLS]")    
    
    mask = [1] * len(tokens)
    input_ids = tokenization.convert_tokens_to_ids(vocab, tokens)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)

    inputids = []
    inputids.append(input_ids)
    masks = []
    masks.append(mask)
    segment_ids = numpy.zeros((1, max_seq_length))
    label_ids = numpy.zeros((1, max_seq_length))
    
    return numpy.asarray(inputids), numpy.asarray(masks), segment_ids, label_ids, tokens    


def get_result(result, tokens, id2label):
    predict = []        
    for i,v in enumerate(result.outputs["output"].int_val):
        label = id2label[v]
        if label != "[CLS]" and label != "[PAD]":
            predict.append( { tokens[i] : label} )
            
    return predict
    
def main(_):
    host = FLAGS.ip_address
    port = FLAGS.port_no
    model_name = FLAGS.model_name
    model_version = -1
    signature_name = ""
    request_timeout = 10.0
    
    with open('middle_data/label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}

    # Generate inference data
    #sentence = "The deal is a major win for Microsoft’s cloud business Azure, which has long been playing catch-up to Amazon’s market-leading Amazon Web Services. Microsoft said it was preparing a statement."
    sentence = FLAGS.sentence
    input_ids, mask, segment_ids, label_ids, tokens = get_feature_vector(sentence)
    input_ids_tensor_proto = tf.contrib.util.make_tensor_proto(input_ids, dtype=tf.int64)
    mask_tensor_proto = tf.contrib.util.make_tensor_proto(mask, dtype=tf.int64)
    label_ids_tensor_proto = tf.contrib.util.make_tensor_proto(label_ids, dtype=tf.int64)
    segment_ids_tensor_proto = tf.contrib.util.make_tensor_proto(segment_ids, dtype=tf.int64)
  
    # Create gRPC client
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    if model_version > 0:
        request.model_spec.version.value = model_version
    if signature_name != "":
        request.model_spec.signature_name = signature_name
    request.inputs["input_ids"].CopyFrom(input_ids_tensor_proto)
    request.inputs["mask"].CopyFrom(mask_tensor_proto)
    request.inputs["label_ids"].CopyFrom(label_ids_tensor_proto)
    request.inputs["segment_ids"].CopyFrom(segment_ids_tensor_proto)
  
    # Send request
    result = stub.Predict(request, request_timeout)
    predict = get_result(result, tokens, id2label)
    print(predict)

if __name__ == "__main__":
    flags.mark_flag_as_required("ip_address")
    flags.mark_flag_as_required("sentence")
    flags.mark_flag_as_required("model_name")
    tf.app.run()