from jean.keyconcepts.entities import filter_annotations


def test_filter_annotations(mocker):
    spacy_entities = mocker.patch('jean.keyconcepts.entities.spacy_entities')
    spacy_entities.return_value = [
        'Nope',
    ]

    annotations = [
        # Exclude for wgt < 0.5
        {
            'title': 'Test 1',
            'wgt': 0.25,
            'support': [{'text': 'test1'}],
            'dbPediaTypes': ['Object']
        },
        # Exclude for Nope in entities
        {
            'title': 'Nope (nope)',
            'wgt': 0.7,
            'support': [{'text': 'Nope'}],
            'dpPediaTypes': ['Person']
        },
        # Exclude for Organisation in dbPediaTypes
        {
            'title': 'Move 37',
            'wgt': 0.8,
            'support': [{'text': 'move37'}],
            'dbPediaTypes': ['Organisation']
        },
        # Include
        {
            'title': 'Awesome',
            'wgt': 1,
            'support': [{'text': 'awesome'}],
            'dbPediaTypes': ['Meme']
        }
    ]

    result = list(filter(filter_annotations('test.com'), annotations))
    assert len(result) == 1
    assert result[0] is annotations[3]
    spacy_entities.assert_called_once_with('test.com')
