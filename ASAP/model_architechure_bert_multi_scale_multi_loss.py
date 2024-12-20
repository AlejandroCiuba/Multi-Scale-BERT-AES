from data import (asap_essay_lengths, 
                  fix_score, )
from document_bert_architectures import (DocumentBertCombineWordDocumentLinear, 
                                         DocumentBertSentenceChunkAttentionLSTM, )
from encoder import encode_documents
from evaluate import evaluation
from tqdm import tqdm
from transformers import (BertConfig, 
                          CONFIG_NAME, 
                          BertTokenizer, )
from typing import (List,
                    Tuple,
                    Union, )

import os
import torch

import torch.nn as nn


class DocumentBertScoringModel(nn.Module):

    def __init__(self, args=None):

        super().__init__()

        if args is not None:
            self.args = vars(args)

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])

        if os.path.exists(self.args['bert_model_path']):

            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))

            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))

            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.args['bert_model_path'])

        self.config = config
        self.prompt = int(args.prompt[1])

        chunk_sizes_str = self.args['chunk_sizes']
        self.chunk_sizes = []
        self.bert_batch_sizes = []

        if "0" != chunk_sizes_str:

            for chunk_size_str in chunk_sizes_str.split("_"):

                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)

        bert_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        print("prompt:%d, asap_essay_length:%d" % (self.prompt, asap_essay_lengths[self.prompt]))
        print("chunk_sizes_str:%s, bert_batch_size_str:%s" % (chunk_sizes_str, bert_batch_size_str))

        self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
            "google-bert/bert-base-uncased", # self.args['bert_model_path'] + "/word_document",
            config=config,
        )

        self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
            "google-bert/bert-base-uncased", # self.args['bert_model_path'] + "/chunk",
            config=config,
        )

    def forward(self, data: List[torch.Tensor], eval: bool=False) -> torch.Tensor:
        """
        Assumes a list of `torch.Tensor` where the first element is the document-level representation,
        and the following are the chunk representations in ascending order. 
        """

        if eval:
            self.bert_regression_by_word_document.eval()
            self.bert_regression_by_chunk.eval()

        predictions = torch.zeros((data[0].shape[0])).to(device=self.args['device'])

        # Get the document-level predictions
        print(data)
        word_document_predictions = self.bert_regression_by_word_document(document_batch=data[0], device=self.args['device'])
        word_document_predictions = torch.squeeze(word_document_predictions)

        assert len(data) - 1 == len(self.chunk_sizes)

        predictions = torch.add(predictions, word_document_predictions)
        # print("Word Document Predictions:", predictions)
        # Add the chunk-level predictions
        for chunk_data, batch_size in zip(data[1:], self.bert_batch_sizes):

            chunk_predictions = self.bert_regression_by_chunk(
                document_batch=chunk_data,
                device=self.args['device'],
                bert_batch_size=batch_size,
            )

            chunk_predictions = torch.squeeze(chunk_predictions)
            # print(f"Batch Size: {batch_size}\nChunk Predictions:", chunk_predictions)

            predictions = torch.add(predictions, chunk_predictions)

        return predictions

    def predict_for_regress(self, data: Tuple[List[str], List[str]]):

        correct_output = None

        if isinstance(data, tuple) and len(data) == 2:

            # Word-level tokenization for the token- and document-scale BERT model
            document_representations_word_document, _ = encode_documents(data[0], 
                                                                         self.bert_tokenizer, 
                                                                         max_input_length=512, )

            # Segment-scale at the various chunk sizes passed in asap.ini
            # List of document token tensors and sequence information per chunk size
            document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []

            for chunk_size in self.chunk_sizes:

                document_representations_chunk, document_sequence_lengths_chunk = encode_documents(data[0],
                                                                                                   self.bert_tokenizer,
                                                                                                   max_input_length=chunk_size, )

                document_representations_chunk_list.append(document_representations_chunk)
                document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)

            correct_output = torch.FloatTensor(data[1])

        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])

        self.bert_regression_by_word_document.eval()
        self.bert_regression_by_chunk.eval()

        # For the token-scale and all segment-scale features, simply add them all together
        with torch.no_grad():

            predictions = torch.zeros((document_representations_word_document.shape[0]))

            for i in tqdm(range(0, document_representations_word_document.shape[0], self.args['batch_size']), desc="Running model..."):

                batch_document_tensors_word_document = document_representations_word_document[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_document_representations_chunk_list = [chunk_rep[i:i + self.args['batch_size']]. \
                                                               to(device=self.args['device']) for chunk_rep in document_representations_chunk_list]

                batch_predictions = self.forward([batch_document_tensors_word_document, *batch_document_representations_chunk_list], eval=True)

                predictions[i:i + self.args['batch_size']] = batch_predictions

            # print(f"My Method [{i}, {i + self.args['batch_size']}]:")
            # print(predictions)

            # predictions = torch.zeros((document_representations_word_document.shape[0]))

            # for i in tqdm(range(0, document_representations_word_document.shape[0], self.args['batch_size']), desc="Running model..."):

            #     batch_document_tensors_word_document = document_representations_word_document[i:i + self.args['batch_size']].to(device=self.args['device'])

            #     batch_predictions_word_document = self.bert_regression_by_word_document(batch_document_tensors_word_document, device=self.args['device'])
            #     batch_predictions_word_document = torch.squeeze(batch_predictions_word_document)

            #     batch_predictions_word_chunk_sentence_doc = batch_predictions_word_document

            #     for chunk_index in range(len(self.chunk_sizes)):

            #         batch_document_tensors_chunk = document_representations_chunk_list[chunk_index][i:i + self.args['batch_size']]. \
            #             to(device=self.args['device'])

            #         batch_predictions_chunk = self.bert_regression_by_chunk(
            #             batch_document_tensors_chunk,
            #             device=self.args['device'],
            #             bert_batch_size=self.bert_batch_sizes[chunk_index],
            #             )

            #         batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
            #         batch_predictions_word_chunk_sentence_doc = torch.add(batch_predictions_word_chunk_sentence_doc, batch_predictions_chunk)
            #         break

            #     predictions[i:i + self.args['batch_size']] = batch_predictions_word_chunk_sentence_doc

            # print(f"Theirs [{i}, {i + self.args['batch_size']}]:")
            # print(predictions)

        assert correct_output.shape == predictions.shape

        predictions = predictions.cpu().numpy()
        prediction_scores = predictions  # [fix_score(item, self.prompt) for item in predictions]
        correct_output = correct_output.cpu().numpy()

        self.to_file(labels=correct_output, predictions=prediction_scores)
        return self.evaluate(labels=correct_output, predictions=prediction_scores)

    def to_file(self, labels: List[Union[int, float]], predictions: List[float]):
        with open(self.args['result_file'], "w") as outfile:
            for pred, label in zip(predictions, labels):
                outfile.write("%f\t%f\n" % (label, pred))

    def evaluate(self, labels: List[Union[int, float]], predictions: List[float]):

        test_eva_res = evaluation(labels, predictions)

        print("pearson:", float(test_eva_res[7]))
        print("qwk:", float(test_eva_res[8]))

        return float(test_eva_res[7]), float(test_eva_res[8])

