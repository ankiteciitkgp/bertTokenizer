package com.ankit.bert.tokenizerimpl;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.ankit.bert.tokenizer.Tokenizer;
import com.ankit.bert.utils.TokenizerUtils;

import lombok.extern.log4j.Log4j2;

/**
 * Constructs a BERT tokenizer. Based on WordPiece.
 * 
 * This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which
 * contains most of the methods. Users should refer to the superclass for more
 * information regarding methods.
 * 
 * Args:
 * 
 * vocab_file (:obj:`string`): File containing the vocabulary.
 * 
 * do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether to
 * lowercase the input when tokenizing.
 * 
 * do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether
 * to do basic tokenization before WordPiece.
 * 
 * never_split (:obj:`bool`, `optional`, defaults to :obj:`True`): List of
 * tokens which will never be split during tokenization. Only has an effect when
 * :obj:`do_basic_tokenize=True`
 * 
 * unk_token (:obj:`string`, `optional`, defaults to "[UNK]"): The unknown
 * token. A token that is not in the vocabulary cannot be converted to an ID and
 * is set to be this token instead.
 * 
 * sep_token (:obj:`string`, `optional`, defaults to "[SEP]"): The separator
 * token, which is used when building a sequence from multiple sequences, e.g.
 * two sequences for sequence classification or for a text and a question for
 * question answering. It is also used as the last token of a sequence built
 * with special tokens.
 * 
 * pad_token (:obj:`string`, `optional`, defaults to "[PAD]"): The token used
 * for padding, for example when batching sequences of different lengths.
 * 
 * cls_token (:obj:`string`, `optional`, defaults to "[CLS]"): The classifier
 * token which is used when doing sequence classification (classification of the
 * whole sequence instead of per-token classification). It is the first token of
 * the sequence when built with special tokens.
 * 
 * mask_token (:obj:`string`, `optional`, defaults to "[MASK]"): The token used
 * for masking values. This is the token used when training this model with
 * masked language modeling. This is the token which the model will try to
 * predict.
 * 
 * tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
 * Whether to tokenize Chinese characters. This should likely be deactivated for
 * Japanese: see: https://github.com/huggingface/transformers/issues/328
 */

@Log4j2
public class BertTokenizer implements Tokenizer {

	private String vocab_file = "vocab.txt";
	private Map<String, Integer> token_id_map;
	private Map<Integer, String> id_token_map;
	private boolean do_lower_case = true;
	private boolean do_basic_tokenize = true;
	private List<String> never_split = new ArrayList<String>();
	private String unk_token = "[UNK]";
	private String sep_token = "[SEP]";
	private String pad_token = "[PAD]";
	private String cls_token = "[CLS]";
	private String mask_token = "[MASK]";
	private boolean tokenize_chinese_chars = true;
	private BasicTokenizer basic_tokenizer;
	private WordpieceTokenizer wordpiece_tokenizer;

	private static final int MAX_LEN = 512;

	public BertTokenizer(String vocab_file, boolean do_lower_case, boolean do_basic_tokenize, List<String> never_split,
			String unk_token, String sep_token, String pad_token, String cls_token, String mask_token,
			boolean tokenize_chinese_chars) {
		this.vocab_file = vocab_file;
		this.do_lower_case = do_lower_case;
		this.do_basic_tokenize = do_basic_tokenize;
		this.never_split = never_split;
		this.unk_token = unk_token;
		this.sep_token = sep_token;
		this.pad_token = pad_token;
		this.cls_token = cls_token;
		this.mask_token = mask_token;
		this.tokenize_chinese_chars = tokenize_chinese_chars;
		init();
	}

	public BertTokenizer() {
		init();
	}

	private void init() {
		try {
			this.token_id_map = load_vocab(vocab_file);
		} catch (IOException e) {
			log.error("Unable to load vocab due to: ", e);
		}
		this.id_token_map = new HashMap<Integer, String>();
		for (String key : token_id_map.keySet()) {
			this.id_token_map.put(token_id_map.get(key), key);
		}

		if (do_basic_tokenize) {
			this.basic_tokenizer = new BasicTokenizer(do_lower_case, never_split, tokenize_chinese_chars);
		}
		this.wordpiece_tokenizer = new WordpieceTokenizer(token_id_map, unk_token);
	}

	private Map<String, Integer> load_vocab(String vocab_file_name) throws IOException {
		ClassLoader classloader = Thread.currentThread().getContextClassLoader();
		InputStream file =classloader.getResourceAsStream(vocab_file_name);
		return TokenizerUtils.generateTokenIdMap(file);
	}

	/**
	 * Tokenizes a piece of text into its word pieces.
	 *
	 * This uses a greedy longest-match-first algorithm to perform tokenization
	 * using the given vocabulary.
	 *
	 * For example: input = "unaffable" output = ["un", "##aff", "##able"]
	 *
	 * Args: text: A single token or whitespace separated tokens. This should have
	 * already been passed through `BasicTokenizer`.
	 *
	 * Returns: A list of wordpiece tokens.
	 * 
	 */
	@Override
	public List<String> tokenize(String text) {
		List<String> split_tokens = new ArrayList<String>();
		if (do_basic_tokenize) {
			for (String token : basic_tokenizer.tokenize(text)) {
				for (String sub_token : wordpiece_tokenizer.tokenize(token)) {
					split_tokens.add(sub_token);
				}
			}
		} else {
			split_tokens = wordpiece_tokenizer.tokenize(text);
		}
		return split_tokens;
	}

	public String convert_tokens_to_string(List<String> tokens) {
		// Converts a sequence of tokens (string) in a single string.
		return tokens.stream().map(s -> s.replace("##", "")).collect(Collectors.joining(" "));
	}

	public List<Integer> convert_tokens_to_ids(List<String> tokens) {
		List<Integer> output = new ArrayList<Integer>();
		for (String s : tokens) {
			output.add(token_id_map.get(s));
		}
		return output;
	}

	public int vocab_size() {
		return token_id_map.size();
	}
}
