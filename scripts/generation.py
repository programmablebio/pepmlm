def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    sequence = protein_seq + binder_seq
    tensor_input = tokenizer.encode(sequence, return_tensors='pt').to(model.device)

    # Create a mask for the binder sequence
    binder_mask = torch.zeros(tensor_input.shape).to(model.device)
    binder_mask[0, -len(binder_seq)-1:-1] = 1

    # Mask the binder sequence in the input and create labels
    masked_input = tensor_input.clone().masked_fill_(binder_mask.bool(), tokenizer.mask_token_id)
    labels = tensor_input.clone().masked_fill_(~binder_mask.bool(), -100)

    with torch.no_grad():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


def generate_peptide_for_single_sequence(protein_seq, peptide_length = 15, top_k = 3, num_binders = 4):

    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    binders_with_ppl = []

    for _ in range(num_binders):
        # Generate binder
        masked_peptide = '<mask>' * peptide_length
        input_sequence = protein_seq + masked_peptide
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]

        # Apply top-k sampling
        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_indices = Categorical(probabilities).sample()
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')

        # Compute PPL for the generated binder
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)

        # Add the generated binder and its PPL to the results list
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl

def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4):
    if isinstance(input_seqs, str):  # Single sequence
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
        return pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])

    elif isinstance(input_seqs, list):  # List of sequences
        results = []
        for seq in input_seqs:
            binders = generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
            for binder, ppl in binders:
                results.append([seq, binder, ppl])
        return pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])