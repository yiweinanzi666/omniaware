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


# Image ids of the question's own images; webpage images use 900000 + j (pipeline ids, see arms.py).
QUERY_IMAGE_ID_BASE = 800000


def official_query(qid, question, sources, query_images=()):
    """One selected source becomes one official webpage, in the fixed order.

    The question's own images (``query_images``, OK-VQA) are passed as the official auxiliary images:
    valid, unscored, and kept by the summarizer's default filters.
    """
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
    aux_images = [dict(image_id=QUERY_IMAGE_ID_BASE + j, split_id=0, valid=True, final_score=0,
        image_url=path, cached_image_url=path, detailed_image_caption='')
        for j, path in enumerate(query_images)]
    return dict(query_id=qid, content=question, webpages=webpages, aux_images=aux_images)


def extractor_messages(question, result, query_images=()):
    """Read the generated document, its captions and retained images, and the question's own images."""
    pieces = result['placeholder_response'].split('<IMAGE_PLACEHOLDER>')
    images = result['output_images']
    if len(pieces) != len(images) + 1:
        raise ValueError('Generated document placeholders and retained images do not match')
    messages, pending = [], EXTRACTOR_INSTRUCTION + '\n\nQuestion: ' + question + '\n\n'
    for path in query_images:
        pending += 'Question image:\n'
        messages += [dict(type='text', text=pending), dict(type='image_url', image_url=dict(url=path))]
        pending = '\n'
    messages.append(dict(type='text', text=pending + 'Generated response:\n' + pieces[0]))
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
