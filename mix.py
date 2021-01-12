import pandas as pd


columns = ["article_id", "start_position", "end_position", "entity_text", "entity_type"]

eric_df = pd.read_csv('./Human_Data/roberta_128_ori_combine_fix_0357.tsv', sep='\t')
hongyun_df = pd.read_csv('./result/dev_1_add_overlap.tsv', sep='\t')

hongyun_df = hongyun_df[hongyun_df['entity_type'] != 'med_exam']
eric_med_exam = eric_df[eric_df['entity_type'] == 'med_exam']

hongyun_df = hongyun_df.append(eric_med_exam, ignore_index=True)
print(hongyun_df)
hongyun_df.to_csv('./result/mix.tsv', index=False, sep="\t")
