"""Shared WebQA evidence and answer interface for the native M2RAG stages."""

EXTRACTOR_INSTRUCTION = (
    'Answer the question concisely using only the supplied generated response '
    'and any images included in that response. Preserve every requested entity, '
    'quantity, comparison, and qualification. If the response already gives a '
    'concise answer, copy its answer. Do not add facts that are absent from the '
    'supplied response. Return only the answer. If the response does not answer '
    'the question, return "Insufficient information".'
)


def prepare_sources(tokenizer, ordered_sources, image_paths, relevance_by_source):
    """Apply the same 256-text-token limit before either generation interface."""
    result = []
    for source in ordered_sources:
        tokens = tokenizer.encode(source['text'], add_special_tokens=False)
        source_id = source['source_id']
        row = dict(source, text=tokenizer.decode(tokens[:256]),
            full_text_tokens=len(tokens), kept_text_token_ids=tokens[:256],
            r_j=relevance_by_source[source_id])
        if source['modality'] == 'image':
            row['image_path'] = image_paths[str(source['fact_id'])]
        result.append(row)
    return result


def official_query(qid, question, sources):
    """One selected source becomes one official webpage, in the fixed order."""
    webpages = []
    for source in sources:
        text = source['text']
        images = []
        if source['modality'] == 'image':
            image_id = int(source['fact_id'])
            text += f'\n<IMAGE_PLACEHOLDER>[{image_id}]\n'
            images.append(dict(image_id=image_id, split_id=0, valid=True, final_score=0,
                image_url=source['image_path'], cached_image_url=source['image_path'],
                detailed_image_caption=source['text']))
        webpages.append(dict(webpage_id=source['source_id'], title=source['source_id'],
            url='', cleaned_webpage_splits=[dict(split_id=0, score=0, text=text)], images=images))
    return dict(query_id=qid, content=question, webpages=webpages, aux_images=[])


def extractor_messages(question, result):
    """Read only the generated document, its captions, and its retained images."""
    pieces = result['placeholder_response'].split('<IMAGE_PLACEHOLDER>')
    images = result['output_images']
    if len(pieces) != len(images) + 1:
        raise ValueError('Generated document placeholders and retained images do not match')
    messages = [dict(type='text', text=EXTRACTOR_INSTRUCTION +
        '\n\nQuestion: ' + question + '\n\nGenerated response:\n' + pieces[0])]
    for index, image in enumerate(images):
        messages.extend([
            dict(type='text', text='\nFigure caption: ' + image['image_caption'] + '\n'),
            dict(type='image_url', image_url=dict(url=image['cached_image_url'])),
            dict(type='text', text=pieces[index+1]),
        ])
    return messages


def direct_answer_document(answer):
    """The direct-QA arm passes its actual answer through the same extractor."""
    return dict(placeholder_response=answer, output_images=[])
