import pandas as pd
import numpy as np
from pathlib import Path
from exp_reader import ExpReader
from metric_extractor import StrokeMetrics, StrokeExtractor, GenMetrics

def run_agg_metrics(df, output_dir, num_buckets):
    gen_dict_names = ['exp_dir','Procedure Duration (s)', 'Bone Voxels Removed', 'Non-Bone Voxels Removed', '%Time with 6mm Burr (%)', '%Time with 4mm Burr (%)',
                      'Number of Strokes']
    stroke_dict_names = ['exp_dir', 'Number of Strokes in Bucket', 'Avg Voxels Removed per Stroke', 'Avg Stroke Length (m)', 'Avg Stroke Velocity (m/s)', 
                      'Avg Stroke Acceleration (m/s^2)', 'Avg Stroke Jerk (m/s^3)', 'Avg Stroke Curvature', 'Avg Stroke Force (N)',
                      'Avg Stroke Drill Angle wrt Camera (degrees)', 'Avg Stroke Drill Angle wrt Bone (degrees)']
    
    agg_gen_dict = {
        'exp_dir': [],
        'procedure_time': [],
        'vxls_removed_bone': [],
        'vxls_removed_non_bone': [],
        '6mm_burr_time': [],
        '4mm_burr_time': [],
        'num_strokes': [],
    }
    
    agg_stroke_dict = {}
    for i in range(num_buckets):
        agg_stroke_dict['bucket_' + str(i)] = {'exp_dir': [],
                                        'num_strokes_b': [],
                                        'vxls_removed': [],
                                        'length': [],
                                        'velocity': [],
                                        'acceleration': [],
                                        'jerk': [],
                                        'curvature': [],
                                        'force': [],
                                        'angle_wrt_camera': [],
                                        'angle_wrt_bone': []}
        
    for i, exp_dir in enumerate(df['exp_dir']):
        exp = ExpReader(exp_dir)
        stroke_extr = StrokeExtractor(exp)
        stroke_metr = StrokeMetrics(stroke_extr, num_buckets = num_buckets)
        stroke_metr_dict = stroke_metr.calc_metrics()
        gen_metr = GenMetrics(stroke_extr, exp_dir)
        gen_metr_dict = gen_metr.calc_metrics()

        print("Processing " + Path(exp_dir).name)
        for i in range(num_buckets):
            agg_stroke_dict['bucket_' + str(i)]['exp_dir'].append(exp_dir)
            
            num_strokes_b = len(np.where(stroke_metr.bucket_dict['bucket_assignments'] == i)[0])
            agg_stroke_dict['bucket_' + str(i)]['num_strokes_b'].append(num_strokes_b)

            for key in agg_stroke_dict['bucket_0'].keys():
                if key == 'num_strokes_b' or key == 'exp_dir':
                    continue
                if key in stroke_metr_dict.keys():
                    agg_stroke_dict['bucket_' + str(i)][key].append(np.average(stroke_metr_dict[key][np.where(stroke_metr.bucket_dict['bucket_assignments'] == i)]))
                else:
                    agg_stroke_dict['bucket_' + str(i)][key].append(np.NaN)
        
        agg_gen_dict['exp_dir'].append(exp_dir)
        agg_gen_dict['procedure_time'].append(gen_metr_dict['procedure_time'])
        agg_gen_dict['num_strokes'].append(gen_metr_dict['num_strokes'])
        
        non_bone_voxels_rmvd = sum(value for key, value in gen_metr_dict['vxl_rmvd_dict'].items() if key != 'Bone')
        agg_gen_dict['vxls_removed_non_bone'].append(non_bone_voxels_rmvd)
        bone_voxels_rmvd = gen_metr_dict['vxl_rmvd_dict']['Bone']
        agg_gen_dict['vxls_removed_bone'].append(bone_voxels_rmvd)
        
        # Sometimes people change burr before the first stroke so they don't start 
        # with a 6mm burr, making it such tha 6mm is not in dictionary
        if '6 mm' in gen_metr_dict['burr_chg_dict']:
            six_mm_burr_time = gen_metr_dict['burr_chg_dict']['6 mm']
        else:
            six_mm_burr_time = 0
        agg_gen_dict['6mm_burr_time'].append(six_mm_burr_time)


        if '4 mm' in gen_metr_dict['burr_chg_dict']:
            four_mm_burr_time = gen_metr_dict['burr_chg_dict']['4 mm']
        else:
            four_mm_burr_time = 0
        agg_gen_dict['4mm_burr_time'].append(four_mm_burr_time)

    
    # print(agg_gen_dict)
    # print(agg_stroke_dict)

    # agg_dict = {**agg_gen_dict, **agg_stroke_dict}
    # agg_df = pd.DataFrame(agg_dict)
    # agg_df.columns = agg_dict_names
    
    # comb = pd.merge(df, agg_df, on='exp_dir', how='inner')
    # comb.to_csv(output_f, index=False)

    agg_gen_df = pd.DataFrame(agg_gen_dict)
    agg_gen_df.columns = gen_dict_names
    combined_gen_df = pd.merge(df, agg_gen_df, on='exp_dir', how='inner')
    combined_gen_df.to_csv(output_dir / 'gen_metrics.csv', index=False)
    
    # Save each bucket's stroke metrics to separate CSV files
    for bucket, data in agg_stroke_dict.items():
 
        bucket_df = pd.DataFrame(data)
        bucket_df.columns =  stroke_dict_names
        bucket_df = pd.merge(df, bucket_df, on='exp_dir', how='inner')
        # bucket_filename = output_dir / f"{bucket}_metrics.csv"
        bucket_filename = output_dir / f"stroke_metrics.csv"
        bucket_df.to_csv(bucket_filename, index=False)
        print(f"Saved bucket data to {bucket_filename}")

def main():
    output_dir = '../output_SDF'
    exp_csv = "/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs_DONOTOVERWRITE.csv"
    num_buckets = 1
    df = pd.read_csv(exp_csv)
    df['expert'] = np.where(df['level_training'] >= 11, 1, 0)
    df = df[(df['participant'] > 3) & (df['assist_mode'] == 'baseline')]

    run_agg_metrics(df, Path(output_dir), num_buckets)


    


if __name__ == "__main__":
    main()