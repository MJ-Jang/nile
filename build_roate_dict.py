import numpy as np
import os
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--src_folder_path',
                    type=str,
                    default='retrieve_knowledge',
                    help='source folder path')

parser.add_argument('--emb_folder_path',
                    type=str,
                    default='retrieve_knowledge',
                    help='source folder path')

parser.add_argument('--save_path',
                    type=str,
                    default='retrieve_knowledge',
                    help='save_path')


def load_files(args):
    # load entity-synset
    entity2synset = {}
    loadFile = open(os.path.join(args.src_folder_path, 'wordnet-mlj12-definitions.txt'))
    for line in loadFile:
        info = line.strip().split('\t')
        offset_str, synset_name = info[0], info[1]
        entity2synset[offset_str] = synset_name

    # load entities dict file
    loadFile = open(os.path.join(args.src_folder_path, 'entities.dict'))
    entity_info= dict()
    for line in loadFile:
        eid, entity = line.strip().split('\t')
        entity_info[entity] = {
            'id': int(eid),
            'synset': entity2synset[entity]
        }

    # load entity embeddings
    entity_embs = np.load(os.path.join(args.emb_folder_path, 'entity_embedding.npy'))
    dim = np.shape(entity_embs)[1]

    outp = {
        'entity_embs': entity_embs,
        'entity_info': entity_info
    }
    return outp, dim


def build_file(entity_embs, entity_info):
    outp = ''
    for key in tqdm(entity_info, desc='Building model'):
        id = entity_info[key]['id']
        synset = entity_info[key]['synset']
        vec = entity_embs[int(id)]

        vec_txt = ' '.join([str(s) for s in list(vec)])
        line = f"{synset} {str(key)} {str(id)} {vec_txt}"
        outp += line + '\n'
    return outp.strip()


def main(args):
    # 1) load files
    outp, dim = load_files(args)

    # 2) build model.txt file
    model = build_file(**outp)

    # 3) save
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Vector dim: {dim}")
    saveFile = open(os.path.join(args.save_path, f'RotatE_Wn18_{dim}d.txt'), 'w', encoding='utf-8')
    saveFile.write(model)


if __name__ == '__main__':
    main(parser.parse_args())


