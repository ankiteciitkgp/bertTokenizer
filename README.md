# Bert Tokenizer

This repository contains java implementation of Bert Tokenizer. The implementation is referred from the Hugging face Transformers library.

https://huggingface.co/transformers/main_classes/tokenizer.html

##Usage

To get tokens from text:
```
String text = "Text to tokenize";
BertTokenizer tokenizer = new BertTokenizer();
List<String> tokens = tokenizer.tokenize(text);
```

To get token ids using Bert Vocab:

```
List<Integer> token_ids = tokenizer.convert_tokens_to_ids(tokens);
```
