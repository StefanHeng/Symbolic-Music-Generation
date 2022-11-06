__all__ = ['sample_full_midi', 'sample_full_step', 'gen_broken']

# `平凡之路
sample_full_midi = 'TimeSig_4/4 Tempo_120 <bar> <melody> p_7/2 d_1 p_2/4 d_1/2 p_10/3 d_1/2 p_3/2 d_1 p_3/4 d_1/2 p_10/3 ' \
              'd_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_10/2 d_1/2 p_5/3 d_1/2 p_2/4 d_1/2 p_10/3 d_1/2 ' \
              'p_9/3 d_1/2 p_10/3 d_1/2 p_12/3 d_1/2 p_5/3 d_1/2 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_7/2 d_1 '\
              'p_2/4 d_1/2 p_10/3 d_1/2 p_3/3 d_1 p_3/4 d_1/2 p_10/3 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> ' \
              'p_10/2 d_1/2 p_5/3 d_1/2 p_2/4 d_1/2 p_10/3 d_1/2 p_9/3 d_1/2 p_10/3 d_1/2 p_12/3 d_1/2 p_5/3 d_1/2 ' \
              '<bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_7/2 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_7/5 d_1/4 p_7/5 d_1/4 '\
              'p_7/5 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 ' \
              '<bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_7/5 d_1/2 p_7/5 d_3/4 '\
              'p_5/5 d_1/4 p_5/5 d_3/4 p_3/5 d_1/4 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 ' \
              'd_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1 p_7/5 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 ' \
              'p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> ' \
              '<melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_10/4 d_1/4 p_3/5 d_1/4 p_3/5 d_1/2 p_3/5 d_1/2 p_3/5 ' \
              'd_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_10/4 d_4 <bass> p_10/2 d_2 p_5/2 d_2 ' \
              '<bar> <melody> p_10/4 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_7/5 d_1/4 p_7/5 d_1/4 p_7/5 d_1/2 p_10/4 d_1/2 ' \
              'p_12/4 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 ' \
              'd_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_7/5 d_1/2 p_7/5 d_3/4 p_5/5 d_1/4 p_5/5 d_3/4 '\
              'p_3/5 d_1/4 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> ' \
              '<melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1 p_7/5 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 p_2/5 d_1/2 <bass> ' \
              'p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 ' \
              'p_2/5 d_1/2 p_2/5 d_1/2 p_10/4 d_1/2 p_3/5 d_1/2 p_3/5 d_1/4 p_3/5 d_1/4 p_3/5 d_1/2 p_2/5 d_1/2 ' \
              '<bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_10/4 d_2 p_10/4 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 ' \
              '<bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 ' \
              'd_1/2 p_7/5 d_1 p_5/5 d_1/2 p_3/5 d_1/4 p_2/5 d_1/4 <bass> p_7/2 d_1 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 ' \
              'd_1 p_10/3 d_1/2 p_3/2 d_1/2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/4 p_12/4 ' \
              'd_1/4 p_12/4 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> ' \
              'p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 d_1/2 p_7/5 d_1 p_7/5 d_1/2 p_10/5 d_1/4 ' \
              'p_10/5 d_1/4 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 ' \
              'd_1/2 p_12/5 d_3/4 p_5/5 d_1/4 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_5/3 d_1/2 p_10/2 d_1/2 ' \
              'p_5/2 d_2 <bar> <melody> p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 p_2/6 d_1/2 p_7/6 d_3/4 p_5/6 d_1/4 ' \
              'p_5/6 d_1/2 p_3/6 d_1/2 <bass> p_7/2 d_1 p_2/3 d_1/2 p_7/2 d_1/2 p_3/2 d_2 <bar> <melody> p_2/6 d_1 ' \
              'p_2/6 d_1/2 p_12/5 d_1/2 p_12/5 d_1 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_5/3 d_1/2 p_10/2 d_1/2 '\
              'p_5/2 d_2 <bar> <melody> p_10/5 d_1 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_3/4 p_10/5 d_1/4 p_9/5 d_1/2 ' \
              'p_10/5 d_1/2 <bass> p_7/2 d_1 p_2/3 d_1/2 p_7/2 d_1/2 p_3/2 d_2 <bar> <melody> p_12/5 d_1 p_10/5 d_1 ' \
              'p_10/5 d_1 p_5/4 d_1 <bass> p_5/2 d_2 p_10/2 d_2 <bar> <melody> p_7/2 d_1 p_2/4 d_1/2 p_10/3 d_1/2 ' \
              'p_3/3 d_1 p_3/4 d_1/2 p_10/3 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_10/2 d_1/2 p_5/3 d_1/2 ' \
              'p_2/4 d_1/2 p_10/3 d_1/2 p_9/3 d_1/2 p_10/3 d_1/2 p_12/3 d_1/2 p_5/3 d_1/2 <bass> p_10/2 d_2 p_5/2 d_2 '\
              '<bar> <melody> p_7/2 d_1 p_2/4 d_1/2 p_10/3 d_1/2 p_3/3 d_1 p_3/4 d_1/2 p_10/3 d_1/2 <bass> p_7/2 d_2 ' \
              'p_3/2 d_2 <bar> <melody> p_10/2 d_1/2 p_5/3 d_1/2 p_2/4 d_1/2 p_10/3 d_1/2 p_9/3 d_1/2 p_10/3 d_1/2 ' \
              'p_12/3 d_1/2 p_5/3 d_1/2 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_7/2 d_1/2 p_2/5 d_1/2 p_2/5 ' \
              'd_1/2 p_7/5 d_1/4 p_7/5 d_1/4 p_7/5 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 '\
              'd_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 ' \
              'd_1/2 p_7/5 d_1/2 p_7/5 d_3/4 p_5/5 d_1/4 p_5/5 d_3/4 p_3/5 d_1/4 <bass> p_7/2 d_2 p_3/2 d_2 <bar> ' \
              '<melody> p_2/5 d_4 <bass> p_10/2 d_1 p_5/3 d_1/2 p_10/2 d_1/2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 ' \
              'p_2/5 d_1/4 p_2/5 d_1/4 p_2/5 d_1 p_7/5 d_1/2 p_10/4 d_1/4 p_10/4 d_1/4 p_12/4 d_1/2 p_2/5 d_1/2 ' \
              '<bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> p_2/5 d_4 <bass> p_10/2 d_1 p_5/3 d_1/2 p_10/2 d_1/2 p_5/2 ' \
              'd_1 p_9/3 d_1/4 p_5/2 d_1/4 p_9/3 d_1/2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_10/4 ' \
              'd_1/2 p_3/5 d_1/2 p_3/5 d_1/4 p_3/5 d_1/4 p_3/5 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_1 p_10/3 d_1/2 p_7/2 ' \
              'd_1/2 p_3/2 d_1 p_3/2 d_1 <bar> <melody> p_10/4 d_1 p_5/4 d_1 p_5/4 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 ' \
              'p_9/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_1 p_5/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 ' \
              'p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 d_1/2 p_7/5 d_1 p_5/5 d_1/2 p_3/5 d_1/4 p_2/5 d_1/4 <bass> p_7/2 d_1 ' \
              'p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_1 p_10/3 d_1/2 p_3/2 d_1/2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 ' \
              'p_2/5 d_1/2 p_2/5 d_1/4 p_12/4 d_1/4 p_12/4 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 ' \
              'd_3/4 p_10/2 d_5/4 p_5/2 d_1/2 p_5/5 d_1/4 p_5/2 d_1/4 p_9/3 d_1/2 p_5/2 d_1/2 <bar> <melody> p_10/5 ' \
              'd_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 d_1/2 p_7/5 d_1 p_7/5 d_1/2 p_10/5 d_1/4 p_10/5 ' \
              'd_1/4 <bass> p_7/2 d_1 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_1 p_10/3 d_1/2 p_3/2 d_1/2 <bar> <melody> ' \
              'p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/2 p_12/5 d_3/4 p_5/5 d_1/4 p_7/5 d_1/2 p_9/5 d_1/2 ' \
              '<bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_1 p_5/2 d_1 <bar> <melody> p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 ' \
              'p_2/6 d_1/2 p_7/6 d_3/4 p_5/6 d_1/4 p_5/6 d_1/2 p_3/6 d_1/2 <bass> p_7/2 d_1 p_7/2 d_1 p_3/2 d_1 p_3/2 '\
              'd_1 <bar> <melody> p_2/6 d_1 p_2/6 d_1/2 p_12/5 d_1/2 p_12/5 d_1 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 '\
              'd_1/2 p_r d_1/4 p_10/2 d_1/4 p_10/3 d_1/2 p_10/2 d_1/2 p_5/2 d_1 p_5/2 d_1 <bar> <melody> p_10/5 d_3/4 '\
              'p_10/5 d_1/4 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_3/4 p_10/5 d_1/4 p_9/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 '\
              'd_1 p_10/3 d_1 p_3/2 d_1 p_3/2 d_1 <bar> <melody> p_12/5 d_1 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_2 ' \
              '<bass> p_5/2 d_2 p_10/2 d_1 p_10/2 d_1 <bar> <melody> p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 ' \
              'd_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_10/4 d_1/2 p_10/4 d_1/2 p_5/4 d_1/2 <bass> p_10/2 d_1 ' \
              'p_2/4 d_1/2 p_10/3 d_1/2 p_10/3 d_3/4 p_10/3 d_1/4 p_10/3 d_1 <bar> <melody> p_10/4 d_1/2 p_9/4 d_1/2 ' \
              'p_10/4 d_3/4 p_10/4 d_1/4 p_9/4 d_1/4 p_10/4 d_1/2 p_7/4 d_1/4 p_10/4 d_1/2 p_10/4 d_1/2 <bass> p_10/2 '\
              'd_4 <bar> <melody> p_10/4 d_1 p_10/4 d_3/4 p_10/4 d_1/4 p_10/4 d_1/2 p_10/4 d_1/4 p_10/4 d_1/4 p_12/4 ' \
              'd_1/2 p_5/4 d_1/2 <bass> p_10/2 d_4 <bar> <melody> p_10/4 d_1/2 p_9/4 d_1/2 p_10/4 d_3/4 p_10/4 d_1/4 ' \
              'p_12/4 d_1 p_12/4 d_1 <bass> p_10/2 d_4 <bar> <melody> p_12/4 d_1/2 p_9/4 d_1/2 p_10/4 d_3/4 p_10/4 ' \
              'd_1/4 p_10/4 d_1/2 p_10/4 d_1/2 p_10/4 d_1/2 p_5/4 d_1/2 <bass> p_7/2 d_4 <bar> <melody> p_5/4 d_1/2 ' \
              'p_5/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_5/4 d_1/4 p_10/4 d_1/4 p_9/4 d_1/4 p_10/4 d_3/4 p_10/4 d_1 ' \
              '<bass> p_10/2 d_4 <bar> <melody> p_10/4 d_1 p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_12/4 ' \
              'd_1/4 p_2/5 d_1/4 p_3/5 d_1/2 p_2/5 d_1/2 <bass> p_10/2 d_4 <bar> <melody> p_10/4 d_1/2 p_9/4 d_1/4 ' \
              'p_10/4 d_1/4 p_12/4 d_1/2 p_5/4 d_1/4 p_10/4 d_1/4 p_12/4 d_1/4 p_2/5 d_1/4 p_3/5 d_1/2 p_2/5 d_1 ' \
              '<bass> p_10/2 d_4 <bar> <melody> p_2/5 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 ' \
              'd_1/4 p_10/4 d_1/2 p_10/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_5/4 d_1/2 <bass> p_10/2 d_4 <bar> <melody> '\
              'p_10/4 d_1 p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_10/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 ' \
              'p_5/4 d_1/2 <bass> p_10/2 d_4 <bar> <melody> p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_9/4 ' \
              'd_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_10/4 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 <bass> p_10/2 d_4 <bar> ' \
              '<melody> p_12/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_5/4 d_1/4 p_10/4 d_1/4 p_9/4 d_1/4 ' \
              'p_10/4 d_1/4 p_12/4 d_1/2 p_12/4 d_1 <bass> p_7/2 d_4 <bar> <melody> p_12/4 d_1/2 p_12/4 d_1/4 p_2/5 ' \
              'd_1/4 p_2/5 d_1/2 p_12/4 d_1/4 p_2/5 d_1/4 p_2/5 d_1/2 p_2/5 d_1/4 p_3/5 d_1/4 p_7/5 d_1/2 p_5/5 d_1/2 '\
              '<bass> p_10/2 d_4 <bar> <melody> p_5/5 d_1/2 p_5/5 d_1/4 p_7/5 d_1/4 p_5/5 d_1/2 p_3/5 d_1/4 p_2/5 ' \
              'd_1/4 p_3/5 d_1/2 p_2/5 d_1/4 p_12/4 d_1/4 p_2/5 d_1/2 p_10/4 d_1/2 <bass> p_10/2 d_4 <bar> <melody> ' \
              'p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_9/4 d_1/4 p_10/4 d_1/4 p_10/4 d_1/2 p_12/4 d_1/4 ' \
              'p_2/5 d_1/4 p_3/5 d_1/2 p_2/5 d_1/2 <bass> p_10/2 d_4 <bar> <melody> p_2/5 d_1/2 p_10/4 d_1/4 p_12/4 ' \
              'd_1/4 p_2/5 d_1/2 p_5/5 d_1/2 p_12/4 d_1/4 p_2/5 d_1/4 p_3/5 d_1/2 p_2/5 d_1 <bass> p_10/2 d_4 <bar> ' \
              '<melody> p_2/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_1/2 ' \
              '<bass> p_7/2 d_1 p_7/3 d_1/2 p_7/2 d_1/2 p_3/2 d_1 p_3/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 ' \
              'p_10/5 d_1/2 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 <bass> p_10/2 d_1 p_10/2 ' \
              'd_1 p_5/2 d_1/2 p_5/2 d_1/2 p_9/3 d_1/2 p_5/2 d_1/2 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 ' \
              'd_1 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_1 p_3/2 d_1 <bar> ' \
              '<melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 p_9/5 d_1/2 ' \
              'p_10/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_1 p_5/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 '\
              'p_10/5 d_1 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 d_1 p_7/2 d_1 p_3/2 d_1 ' \
              'p_3/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 ' \
              'p_9/5 d_1/2 p_10/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_1 p_5/2 d_1 <bar> <melody> p_10/5 d_1/2 ' \
              'p_10/5 d_1/2 p_10/5 d_1 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 d_1 p_7/2 d_1 '\
              'p_3/2 d_3/4 p_3/2 d_1/4 p_10/3 d_1/2 p_3/2 d_1/2 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 '\
              'p_10/5 d_1/2 p_9/5 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_1 ' \
              'p_5/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 d_1/2 p_7/5 d_1 ' \
              'p_5/5 d_1/2 p_3/5 d_1/4 p_2/5 d_1/4 <bass> p_7/2 d_1 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_3/4 p_3/2 d_1/4 ' \
              'p_3/2 d_1/2 p_3/2 d_1/2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/4 p_12/4 d_1/4 ' \
              'p_12/4 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_2 <bar> <melody> '\
              'p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 d_1/2 p_7/5 d_1 p_7/5 d_1/2 p_10/5 d_1/4 ' \
              'p_10/5 d_1/4 <bass> p_7/2 d_1 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_1 p_3/2 d_1 <bar> <melody> p_10/5 d_1/2 '\
              'p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/4 p_12/5 d_1/4 p_12/5 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 ' \
              '<bass> p_10/2 d_1 p_10/2 d_1/4 p_10/2 d_1/4 p_5/2 d_1/2 p_5/2 d_2 <bar> <melody> p_10/5 d_1/2 p_9/5 ' \
              'd_1/2 p_10/5 d_1/2 p_2/6 d_1/2 p_7/6 d_3/4 p_5/6 d_1/4 p_5/6 d_1/2 p_3/6 d_1/2 <bass> p_7/2 d_1 p_10/3 '\
              'd_1/2 p_7/2 d_1/2 p_3/2 d_1 p_3/2 d_1 <bar> <melody> p_2/6 d_1 p_2/6 d_1/2 p_12/5 d_1/2 p_12/5 d_1 ' \
              'p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_1/2 p_9/3 d_1/4 p_5/2 d_1/4 p_9/3 d_1/2 ' \
              'p_5/2 d_1/2 <bar> <melody> p_10/5 d_3/4 p_10/5 d_1/4 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_3/4 p_10/5 ' \
              'd_1/4 p_9/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 d_1/2 p_r d_1/4 p_7/2 d_1/4 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 '\
              'd_1 p_3/2 d_1 <bar> <melody> p_12/5 d_1 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_2 <bass> p_5/2 d_1 p_5/3 ' \
              'd_1/2 p_5/2 d_1/2 p_10/2 d_1 p_10/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 ' \
              'd_1/4 p_7/5 d_1/2 p_7/5 d_1 p_5/5 d_1/2 p_3/5 d_1/4 p_2/5 d_1/4 <bass> p_7/2 d_1 p_7/2 d_1 p_3/2 d_1 ' \
              'p_3/2 d_1 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/4 p_12/4 d_1/4 p_12/4 d_1/2 ' \
              'p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_3/4 p_10/2 d_1/4 p_10/3 d_1/2 p_10/2 d_1/2 p_5/2 ' \
              'd_3/4 p_5/2 d_1/4 p_9/3 d_1/2 p_5/2 d_1/2 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 ' \
              'd_1/4 p_7/5 d_1/2 p_7/5 d_1 p_7/5 d_1/2 p_10/5 d_1/4 p_10/5 d_1/4 <bass> p_7/2 d_1 p_7/2 d_1 p_3/2 d_1 '\
              'p_3/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/2 p_12/5 d_1/4 p_12/5 d_1/4 p_12/5 d_1/2 '\
              'p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_3/4 p_5/2 d_1/4 p_9/3 d_1/4 ' \
              'p_5/2 d_1/4 p_9/3 d_1/2 <bar> <melody> p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 p_2/6 d_1/2 p_7/6 d_3/4 ' \
              'p_5/6 d_1/4 p_5/6 d_1/2 p_3/6 d_1/2 <bass> p_7/2 d_1 p_10/3 d_1/2 p_5/2 d_1/2 p_3/3 d_3/4 p_3/3 d_1/4 ' \
              'p_10/3 d_1/2 p_3/3 d_1/2 <bar> <melody> p_2/6 d_1 p_2/6 d_1/2 p_12/5 d_1/2 p_12/5 d_1 p_7/5 d_1/2 ' \
              'p_9/5 d_1/2 <bass> p_10/2 d_3/4 p_10/2 d_1/4 p_10/3 d_1/2 p_10/2 d_1/2 p_5/2 d_3/4 p_5/2 d_1/4 p_9/3 ' \
              'd_1/2 p_5/2 d_1/2 <bar> <melody> p_10/5 d_3/4 p_10/5 d_1/4 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_3/4 ' \
              'p_10/5 d_1/4 p_9/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 d_3/4 p_7/2 d_1/4 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 ' \
              'd_1 p_3/2 d_1 <bar> <melody> p_12/5 d_1 p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_2 <bass> p_5/2 d_1 p_5/2 d_1 '\
              'p_10/2 d_1 p_10/2 d_1 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 p_7/5 d_1/2 ' \
              'p_7/5 d_1 p_5/5 d_1/2 p_3/5 d_1/4 p_2/5 d_1/4 <bass> p_7/2 d_1 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_3/4 ' \
              'p_3/2 d_1/4 p_10/3 d_1/2 p_3/2 d_1/2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/4 ' \
              'p_12/4 d_1/4 p_12/4 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> p_10/2 d_1 p_10/2 d_1 p_5/2 d_3/4 '\
              'p_5/2 d_1/4 p_9/3 d_1/2 p_5/2 d_1/2 <bar> <melody> p_10/5 d_1/2 p_10/5 d_1/2 p_10/5 d_1/4 p_5/5 d_1/4 ' \
              'p_7/5 d_1/2 p_7/5 d_1 p_7/5 d_1/2 p_10/5 d_1/4 p_10/5 d_1/4 <bass> p_7/2 d_3/4 p_7/2 d_1/4 p_10/3 ' \
              'd_1/2 p_7/2 d_1/2 p_3/2 d_3/4 p_3/2 d_1/4 p_10/3 d_1/2 p_3/2 d_1/2 <bar> <melody> p_10/5 d_1/2 p_10/5 ' \
              'd_1/2 p_10/5 d_1/2 p_12/5 d_1/4 p_12/5 d_1/4 p_12/5 d_1/2 p_5/5 d_1/2 p_7/5 d_1/2 p_9/5 d_1/2 <bass> ' \
              'p_10/2 d_3/4 p_10/2 d_1/4 p_10/3 d_1/2 p_10/2 d_1/2 p_5/2 d_3/4 p_5/2 d_1/4 p_9/3 d_1/2 p_5/2 d_1/2 ' \
              '<bar> <melody> p_10/5 d_1/2 p_9/5 d_1/2 p_10/5 d_1/2 p_2/6 d_1/2 p_7/6 d_3/4 p_5/6 d_1/4 p_5/6 d_1/2 ' \
              'p_3/6 d_1/2 <bass> p_7/2 d_3/4 p_7/2 d_1/4 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_3/4 p_3/2 d_1/4 p_10/3 ' \
              'd_1/2 p_3/2 d_1/2 <bar> <melody> p_2/6 d_1 p_2/6 d_1/2 p_12/5 d_1/2 p_12/5 d_1 p_7/5 d_1/2 p_9/5 d_1/2 '\
              '<bass> p_10/2 d_3/4 p_10/2 d_1/4 p_10/3 d_1/2 p_10/2 d_1/2 p_5/2 d_2 <bar> <melody> p_10/5 d_3/4 ' \
              'p_10/5 d_1/4 p_10/5 d_1/2 p_12/5 d_1/2 p_10/5 d_3/4 p_10/5 d_1/4 p_9/5 d_1/2 p_10/5 d_1/2 <bass> p_7/2 '\
              'd_1 p_10/3 d_1/2 p_7/2 d_1/2 p_3/2 d_1 p_3/2 d_1 <bar> <melody> p_12/5 d_1 p_10/5 d_1/2 p_9/5 d_1/2 ' \
              'p_10/5 d_2 <bass> p_5/2 d_1 p_5/2 d_1 p_10/2 d_2 <bar> <melody> p_2/3 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 ' \
              'p_7/5 d_1/4 p_7/5 d_1/4 p_7/5 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 ' \
              '<bar> <melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 ' \
              'd_1/2 p_7/5 d_1/2 p_7/5 d_3/4 p_5/5 d_1/4 p_5/5 d_3/4 p_3/5 d_1/4 <bass> p_7/2 d_2 p_3/2 d_2 <bar> ' \
              '<melody> p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/4 p_2/5 d_1/4 ' \
              'p_2/5 d_1 p_7/5 d_1/2 p_10/4 d_1/2 p_12/4 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> ' \
              'p_2/5 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_2/5 d_1/2 p_2/5 d_1/2 p_2/5 d_1/2 p_10/4 d_1/4 ' \
              'p_3/5 d_1/4 p_3/5 d_1/2 p_3/5 d_1/2 p_3/5 d_1/2 p_2/5 d_1/2 <bass> p_7/2 d_2 p_3/2 d_2 <bar> <melody> ' \
              'p_10/4 d_4 <bass> p_10/2 d_2 p_5/2 d_2 <bar> <melody> p_10/4 d_2 p_r d_2 <bass> p_r d_4 </s> '

sample_full_step = 'TimeSig_1/4 Tempo_115 <bar> <melody> p_r d_1/2 p_2/5_C d_1/2 <bass> <tup> p_r p_11/3_B p_11/2_B ' \
                   'd_1 </tup> <bar> <melody> p_2/5_C d_1 <bass> p_11/2_B d_1 <bar> <melody> p_2/5_C d_1 <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_2/5_C d_1 <bass> p_11/3_B d_1/4 p_r d_3/4 <bar> <melody> p_2/5_C ' \
                   'd_1/4 p_10/3_A d_1/4 p_1/5_C d_1/2 <bass> p_6/4_F d_1/4 p_6/2_F d_1/4 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_1/5_C d_1 <bass> p_6/2_F d_1 <bar> <melody> p_1/5_C d_1 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_1/5_C d_1 <bass> p_6/2_F d_1/2 p_r d_1/2 <bar> <melody> p_1/5_C d_1/4 p_r d_1/4 ' \
                   'p_11/4_B d_1/2 <bass> p_r d_1/2 p_7/2_F d_1/2 <bar> <melody> p_11/4_B d_1 <bass> p_7/2_F d_1 ' \
                   '<bar> <melody> p_11/4_B d_1 <bass> p_7/2_F d_1 <bar> <melody> p_11/4_B d_1 <bass> p_7/4_F d_1 ' \
                   '<bar> <melody> p_11/4_B d_1/2 p_11/4_B d_1/2 <bass> <tup> p_r p_6/2_F p_r d_1 </tup> <bar> ' \
                   '<melody> p_11/4_B d_1 <bass> p_4/4_E d_1 <bar> <melody> p_11/4_B d_1/2 p_1/5_C d_1/2 <bass> <tup> ' \
                   'p_r p_6/3_F p_6/3_F d_1 </tup> <bar> <melody> p_6/5_F d_1/4 p_6/5_F d_3/4 <bass> p_6/3_F d_1/4 ' \
                   'p_r d_3/4 <bar> <melody> p_1/5_C d_1/4 p_r d_1/4 p_2/5_C d_1/2 <bass> <tup> p_6/4_F p_11/2_B ' \
                   'p_11/2_B d_1 </tup> <bar> <melody> p_2/5_C d_1 <bass> p_11/2_B d_1 <bar> <melody> p_2/5_C d_1 ' \
                   '<bass> p_11/2_B d_1 <bar> <melody> p_2/5_C d_1 <bass> p_11/3_B d_1/4 p_r d_3/4 <bar> <melody> ' \
                   'p_2/5_C d_1/2 p_1/5_C d_1/2 <bass> p_6/4_F d_1/4 p_6/2_F d_1/4 p_6/2_F d_1/2 <bar> <melody> ' \
                   'p_1/5_C d_1 <bass> p_6/2_F d_1 <bar> <melody> p_1/5_C d_1 <bass> p_6/2_F d_1 <bar> <melody> ' \
                   'p_1/5_C d_1 <bass> p_6/2_F d_1/2 p_r d_1/2 <bar> <melody> p_1/5_C d_1/2 p_11/4_B d_1/2 <bass> ' \
                   '<tup> p_r p_7/3_F p_7/2_F d_1 </tup> <bar> <melody> p_11/4_B d_1 <bass> p_7/2_F d_1 <bar> ' \
                   '<melody> p_11/4_B d_1 <bass> p_7/2_F d_3/4 p_7/2_F d_1/4 <bar> <melody> p_11/4_B d_1 <bass> ' \
                   'p_7/4_F d_1 <bar> <melody> p_11/4_B d_1/4 p_10/3_A d_1/4 p_10/4_A d_1/2 <bass> <tup> p_r p_6/2_F ' \
                   'p_r d_1 </tup> <bar> <melody> p_10/4_A d_1 <bass> p_6/4_F d_1 <bar> <melody> <tup> p_r p_1/5_C ' \
                   'p_6/5_F d_1 </tup> <bass> <tup> p_r p_6/4_F p_6/3_F d_1 </tup> <bar> <melody> p_6/5_F d_1 <bass> ' \
                   'p_6/3_F d_1/4 p_6/4_F d_3/4 <bar> <melody> p_6/5_F d_1/4 p_r d_1/4 p_6/5_F d_1/4 p_r d_1/4 <bass> ' \
                   'p_6/4_F d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_2/5_C d_1 <bass> p_11/2_B d_3/4 p_11/4_B ' \
                   'd_1/4 <bar> <melody> p_2/5_C d_1 <bass> p_11/4_B d_1/2 p_11/2_B d_1/2 <bar> <melody> p_2/5_C d_1 ' \
                   '<bass> p_11/2_B d_1 <bar> <melody> p_2/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/2 <bass> p_11/2_B d_1/4 ' \
                   'p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_1/5_C d_1 <bass> p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> ' \
                   'p_1/5_C d_1 <bass> p_10/4_A d_1/2 p_6/2_F d_1/2 <bar> <melody> p_1/5_C d_1 <bass> p_6/2_F d_1 ' \
                   '<bar> <melody> p_1/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 <bass> p_11/4_B d_1/4 p_r d_1/4 ' \
                   'p_4/2_E d_1/2 <bar> <melody> p_11/4_B d_1 <bass> p_4/2_E d_1 <bar> <melody> p_11/4_B d_1 <bass> ' \
                   'p_7/4_F d_1/2 p_4/2_E d_1/2 <bar> <melody> p_11/4_B d_1 <bass> p_4/2_E d_1 <bar> <melody> ' \
                   'p_11/4_B d_1/4 p_r d_1/4 p_1/5_C d_1/2 <bass> p_4/2_E d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_2/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> ' \
                   'p_1/5_C d_1 <bass> p_10/4_A d_1/2 p_6/2_F d_1/2 <bar> <melody> p_6/5_F d_1 <bass> p_6/2_F d_1 ' \
                   '<bar> <melody> p_4/5_E d_1/4 p_r d_1/4 p_6/5_F d_1/2 <bass> p_6/2_F d_1/4 p_r d_1/4 p_11/2_B ' \
                   'd_1/2 <bar> <melody> p_6/5_F d_1 <bass> p_11/2_B d_1/2 p_11/4_B d_1/2 <bar> <melody> p_6/5_F d_1 ' \
                   '<bass> p_11/4_B d_1/2 p_11/2_B d_1/2 <bar> <melody> p_6/5_F d_1 <bass> p_11/2_B d_1 <bar> ' \
                   '<melody> p_6/5_F d_1/4 p_r d_1/4 p_6/5_F d_1/2 <bass> p_2/5_C d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_6/5_F d_1 <bass> p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1 <bass> p_12/4_B ' \
                   'd_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_6/5_F d_1 <bass> p_6/2_F d_1 <bar> <melody> ' \
                   'p_6/5_F d_1/4 p_r d_1/4 p_2/5_C d_1/2 <bass> p_11/4_B d_1/4 p_r d_1/4 p_4/2_E d_1/2 <bar> ' \
                   '<melody> p_2/5_C d_1 <bass> p_4/2_E d_3/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1 <bass> p_7/4_F ' \
                   'd_1/2 p_4/2_E d_1/2 <bar> <melody> p_2/5_C d_3/4 p_r d_1/4 <bass> p_4/2_E d_1 <bar> <melody> ' \
                   'p_11/4_B d_1/4 p_r d_1/4 p_6/5_F d_1/2 <bass> p_r d_1/2 p_6/2_F d_1/2 <bar> <melody> p_6/5_F d_1 ' \
                   '<bass> p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1 <bass> p_1/5_C d_1/4 p_r d_1/4 p_6/2_F ' \
                   'd_1/2 <bar> <melody> p_6/5_F d_1 <bass> p_6/2_F d_1 <bar> <melody> p_6/5_F d_1/4 p_r d_1/4 ' \
                   'p_11/5_B d_1/2 <bass> p_1/5_C d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/5_B d_3/4 p_r ' \
                   'd_1/4 <bass> p_11/2_B d_1 <bar> <melody> p_2/5_C d_1 <bass> p_11/2_B d_1/4 p_r d_1/4 p_11/2_B ' \
                   'd_1/2 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_1/6_C d_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> ' \
                   '<melody> p_11/5_B d_1/4 p_r d_1/4 p_10/5_A d_1/4 p_r d_1/4 <bass> p_6/4_F d_1/2 p_6/2_F d_1/2 ' \
                   '<bar> <melody> p_6/5_F d_1/4 p_r d_1/4 p_6/5_F d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> ' \
                   'p_4/5_E d_1/4 p_r d_1/4 p_6/5_F d_1/2 <bass> p_6/2_F d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> ' \
                   'p_4/5_E d_1/4 p_r d_1/4 p_6/5_F d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_4/5_E d_1/4 ' \
                   'p_r d_1/4 p_6/5_F d_1/2 <bass> p_6/2_F d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_6/5_F ' \
                   'd_1/2 p_10/5_A d_1/2 <bass> p_11/2_B d_1 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 p_2/4_C d_1/2 ' \
                   '<bass> p_11/2_B d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 p_10/5_A ' \
                   'd_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 p_10/5_A d_1/4 p_r ' \
                   'd_1/4 <bass> p_11/2_B d_1/4 p_r d_1/4 p_1/3_C d_1/2 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 ' \
                   'p_1/6_C d_1/4 p_r d_1/4 <bass> p_1/5_C d_1 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 p_11/5_B d_1/4 ' \
                   'p_r d_1/4 <bass> p_1/5_C d_1/2 p_1/3_C d_1/2 <bar> <melody> p_7/5_F d_1/4 p_r d_1/4 p_r d_1/2 ' \
                   '<bass> p_1/3_C d_1/4 p_r d_3/4 <bar> <melody> p_4/5_E d_1/4 p_r d_1/4 p_6/5_F d_1/2 <bass> p_r ' \
                   'd_1/2 p_11/2_B d_1/2 <bar> <melody> p_6/5_F d_1/2 p_11/5_B d_1/2 <bass> p_11/2_B d_1/2 p_r d_1/2 ' \
                   '<bar> <melody> p_11/5_B d_1/2 p_2/4_C d_1/2 <bass> p_2/5_C d_1/2 p_11/2_B d_1/2 <bar> <melody> ' \
                   'p_11/5_B d_1/4 p_r d_1/4 p_1/6_C d_1/4 p_r d_1/4 <bass> p_11/2_B d_3/4 p_r d_1/4 <bar> <melody> ' \
                   'p_11/5_B d_1/4 p_r d_1/4 p_10/5_A d_1/4 p_r d_1/4 <bass> p_2/5_C d_1/4 p_r d_1/4 p_6/2_F d_1/2 ' \
                   '<bar> <melody> p_6/5_F d_1 <bass> p_1/5_C d_1 <bar> <melody> p_6/5_F d_1 <bass> p_1/5_C d_1/2 ' \
                   'p_6/2_F d_1/2 <bar> <melody> p_6/5_F d_1 <bass> p_6/2_F d_1 <bar> <melody> p_6/5_F d_1/2 p_6/5_F ' \
                   'd_1/4 p_r d_1/4 <bass> p_6/2_F d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/4_B d_3/4 p_r ' \
                   'd_1/4 <bass> p_11/2_B d_1/4 p_r d_3/4 <bar> <melody> p_11/4_B d_1/4 p_r d_1/4 p_2/5_C d_1/2 ' \
                   '<bass> p_r d_1/2 p_11/2_B d_1/2 <bar> <melody> p_2/5_C d_1 <bass> p_11/2_B d_1 <bar> <melody> ' \
                   'p_1/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/2 <bass> p_11/4_B d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_1/5_C d_1 <bass> p_6/2_F d_1/2 p_r d_1/2 <bar> <melody> p_1/5_C d_1/4 p_r d_1/4 ' \
                   'p_11/4_B d_1/2 <bass> p_10/4_A d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_10/4_A d_1/2 ' \
                   'p_6/5_F d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_5/5_E d_1/4 p_r d_1/4 p_6/5_F d_1/4 ' \
                   'p_r d_1/4 <bass> p_6/2_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1 ' \
                   '<bass> p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1 <bass> ' \
                   'p_11/4_B d_1/2 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 ' \
                   'p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/2 <bass> p_11/4_B ' \
                   'd_1/4 p_r d_1/4 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_1/5_C d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 ' \
                   'p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_1/5_C d_1 <bass> p_10/4_A d_1/2 p_6/2_F d_1/4 p_r d_1/4 ' \
                   '<bar> <melody> p_1/5_C d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> ' \
                   'p_1/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 <bass> p_11/4_B d_1/4 p_r d_1/4 p_4/2_E d_1/4 p_r ' \
                   'd_1/4 <bar> <melody> p_11/4_B d_1 <bass> p_4/3_E d_1/4 p_r d_1/4 p_4/2_E d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_11/4_B d_1 <bass> p_7/4_F d_1/2 p_4/2_E d_1/2 <bar> <melody> p_11/4_B d_1 <bass> ' \
                   'p_4/3_E d_1/4 p_r d_1/4 p_4/2_E d_1/4 p_r d_1/4 <bar> <melody> p_11/4_B d_1/4 p_r d_1/4 p_2/5_C ' \
                   'd_1/4 p_r d_1/4 <bass> p_7/4_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C ' \
                   'd_1/4 p_r d_1/4 p_2/5_C d_1/4 p_r d_1/4 <bass> p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 ' \
                   '<bar> <melody> p_1/5_C d_1/2 p_2/5_C d_1/4 p_r d_1/4 <bass> p_10/4_A d_1/2 p_11/2_B d_1/4 p_r ' \
                   'd_1/4 <bar> <melody> p_6/5_F d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_4/5_E d_1/4 p_r d_1/4 p_6/5_F d_1/2 <bass> p_r d_1/2 p_11/2_B d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_6/5_F d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> ' \
                   'p_6/5_F d_1 <bass> p_11/4_B d_1/2 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1 <bass> ' \
                   'p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1/4 p_r d_1/4 p_6/5_F ' \
                   'd_1/2 <bass> p_6/3_F d_1/4 p_r d_1/4 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1 <bass> ' \
                   'p_6/3_F d_1/4 p_r d_1/4 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1 <bass> p_1/5_C d_1/2 ' \
                   'p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_6/5_F d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 p_6/2_F d_1/4 ' \
                   'p_r d_1/4 <bar> <melody> p_6/5_F d_1/4 p_r d_1/4 p_2/5_C d_1/2 <bass> p_11/4_B d_1/4 p_r d_1/4 ' \
                   'p_4/2_E d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1 <bass> p_4/3_E d_1/4 p_r d_1/4 p_4/2_E d_1/4 ' \
                   'p_r d_1/4 <bar> <melody> p_2/5_C d_1 <bass> p_7/4_F d_1/2 p_4/2_E d_1/4 p_r d_1/4 <bar> <melody> ' \
                   'p_2/5_C d_3/4 p_r d_1/4 <bass> p_4/3_E d_1/4 p_r d_1/4 p_4/2_E d_1/2 <bar> <melody> p_11/4_B ' \
                   'd_1/4 p_r d_1/4 p_4/5_E d_1/2 <bass> p_4/3_E d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_2/5_C d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 <bar> <melody> ' \
                   'p_1/5_C d_1/2 p_2/5_C d_1/4 p_r d_1/4 <bass> p_1/5_C d_1/4 p_r d_1/4 p_11/2_B d_1/4 p_r d_1/4 ' \
                   '<bar> <melody> p_1/5_C d_1 <bass> p_6/3_F d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_2/5_C ' \
                   'd_1/4 p_r d_1/4 p_2/6_C d_1/2 <bass> p_6/3_F d_1/4 p_r d_1/4 p_7/2_F d_1/2 <bar> <melody> p_2/6_C ' \
                   'd_1 <bass> p_7/2_F d_1 <bar> <melody> p_2/6_C d_1/2 p_11/5_B d_1/2 <bass> p_7/2_F d_1 <bar> ' \
                   '<melody> p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> p_7/2_F d_3/4 p_r d_1/4 <bar> <melody> ' \
                   'p_11/5_B d_1/4 p_r d_1/4 p_9/5_G d_1/2 <bass> p_11/3_B d_1/4 p_r d_1/4 p_2/3_C d_1/2 <bar> ' \
                   '<melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> ' \
                   '<melody> p_9/5_G d_3/4 p_r d_1/4 <bass> p_2/3_C d_3/4 p_r d_1/4 <bar> <melody> p_4/5_E d_1/4 p_r ' \
                   'd_1/4 p_10/5_A d_1/2 <bass> p_r d_1/2 p_6/2_F d_1/2 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F ' \
                   'd_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F ' \
                   'd_1/4 p_r d_1/4 p_6/4_F d_1/2 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 p_11/5_B d_1/2 <bass> ' \
                   'p_1/4_C d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 ' \
                   '<bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 ' \
                   '<bass> p_11/2_B d_1/4 p_r d_1/4 p_11/3_B d_1/4 p_r d_1/4 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 ' \
                   'p_2/6_C d_1/2 <bass> p_11/3_B d_1/4 p_r d_1/4 p_7/2_F d_1/2 <bar> <melody> p_2/6_C d_1 <bass> ' \
                   'p_7/2_F d_1 <bar> <melody> p_2/6_C d_1/2 p_11/5_B d_1/2 <bass> p_7/2_F d_1 <bar> <melody> ' \
                   'p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> p_7/2_F d_1/2 p_2/4_C d_1/2 <bar> <melody> p_11/5_B ' \
                   'd_1/4 p_r d_1/4 p_9/5_G d_1/2 <bass> p_11/3_B d_1/4 p_r d_1/4 p_2/3_C d_1/2 <bar> <melody> ' \
                   'p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> ' \
                   'p_9/5_G d_3/4 p_6/4_F d_1/4 <bass> p_2/3_C d_1/4 p_r d_1/4 p_9/3_G d_1/4 p_r d_1/4 <bar> <melody> ' \
                   'p_4/5_E d_1/4 p_r d_1/4 p_10/5_A d_1/2 <bass> p_9/3_G d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_10/5_A d_1 <bass> p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 ' \
                   'p_11/5_B d_1/2 <bass> p_1/4_C d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/5_B d_1 <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_11/6_B d_1/2 <bass> p_11/2_B d_1/2 ' \
                   'p_11/2_B d_1/4 p_6/6_F d_1/4 <bar> <melody> p_6/6_F d_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_6/6_F ' \
                   'd_1/4 <bass> p_11/3_B d_1/4 p_r d_1/4 p_2/4_C d_1/4 p_r d_1/4 <bar> <melody> p_2/6_C d_1/4 ' \
                   'p_11/5_B d_1/4 p_11/6_B d_1/4 p_6/6_F d_1/4 <bass> p_11/3_B d_1/4 p_r d_1/4 p_2/4_C d_1/4 p_r ' \
                   'd_1/4 <bar> <melody> p_6/6_F d_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_6/6_F d_1/4 <bass> p_11/3_B ' \
                   'd_1/4 p_r d_1/4 p_2/4_C d_1/2 <bar> <melody> p_6/6_F d_1/4 p_11/5_B d_1/4 p_10/6_A d_1/4 p_6/6_F ' \
                   'd_1/4 <bass> p_11/3_B d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_10/6_A d_1/4 p_6/6_F d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_10/6_A d_1/4 p_6/6_F d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_10/6_A d_1/4 p_6/6_F d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_6/2_F d_1/4 p_r d_1/4 p_4/2_E d_1/2 <bar> <melody> p_2/6_C ' \
                   'd_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_4/2_E d_1 <bar> <melody> p_2/6_C d_1/4 ' \
                   'p_11/5_B d_1/4 p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_4/2_E d_1 <bar> <melody> p_2/6_C d_1/4 ' \
                   'p_11/5_B d_1/4 p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_4/2_E d_1/2 p_4/4_E d_1/2 <bar> <melody> ' \
                   'p_2/6_C d_1/4 p_11/5_B d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_11/3_B d_1/4 p_r d_1/4 p_6/2_F ' \
                   'd_1/2 <bar> <melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_6/2_F d_1 ' \
                   '<bar> <melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_6/2_F d_3/4 p_10/3_A ' \
                   'd_1/4 <bar> <melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_11/6_B d_1/2 <bass> p_6/3_F d_1/4 p_r d_1/4 ' \
                   'p_11/2_B d_1/2 <bar> <melody> p_6/6_F d_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_6/6_F d_1/4 <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_2/6_C d_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_6/6_F d_1/4 <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_6/6_F d_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_6/6_F d_1/4 <bass> ' \
                   'p_11/2_B d_1/4 p_r d_1/4 p_2/4_C d_1/2 <bar> <melody> p_6/6_F d_1/4 p_11/5_B d_1/4 p_10/6_A d_1/4 ' \
                   'p_6/6_F d_1/4 <bass> p_11/3_B d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_1/6_C d_1/4 p_10/5_A ' \
                   'd_1/4 p_10/6_A d_1/4 p_6/6_F d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_10/6_A d_1/4 p_6/6_F d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_10/6_A d_1/4 p_6/6_F d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_10/5_A d_1/4 ' \
                   'p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_6/2_F d_1/4 p_r d_1/4 p_4/2_E d_1/2 <bar> <melody> p_2/6_C ' \
                   'd_1/4 p_11/5_B d_1/4 p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_4/2_E d_1 <bar> <melody> p_2/6_C d_1/4 ' \
                   'p_11/5_B d_1/4 p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_4/2_E d_1 <bar> <melody> p_2/6_C d_1/4 ' \
                   'p_11/5_B d_1/4 p_11/6_B d_1/4 p_7/6_F d_1/4 <bass> p_4/2_E d_1/2 p_4/4_E d_1/2 <bar> <melody> ' \
                   'p_2/6_C d_1/4 p_11/5_B d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_11/3_B d_1/4 p_r d_1/4 p_6/2_F ' \
                   'd_1/2 <bar> <melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_6/2_F d_1 ' \
                   '<bar> <melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_1/7_C d_1/4 p_10/6_A d_1/4 <bass> p_6/2_F d_3/4 p_10/3_A ' \
                   'd_1/4 <bar> <melody> p_6/6_F d_1/4 p_1/6_C d_1/4 p_2/6_C d_1/2 <bass> p_6/3_F d_1/4 p_r d_1/4 ' \
                   'p_7/2_F d_1/2 <bar> <melody> p_2/6_C d_1 <bass> p_7/2_F d_1 <bar> <melody> p_2/6_C d_1/2 p_11/5_B ' \
                   'd_1/2 <bass> p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> p_7/2_F ' \
                   'd_3/4 p_r d_1/4 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_10/5_A d_1/4 p_r d_1/4 <bass> p_11/3_B ' \
                   'd_1/4 p_r d_1/4 p_2/3_C d_1/2 <bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> ' \
                   'p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_3/4 p_r d_1/4 <bass> p_2/3_C d_3/4 p_r ' \
                   'd_1/4 <bar> <melody> p_4/5_E d_1/4 p_r d_1/4 p_10/5_A d_1/2 <bass> p_r d_1/2 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_10/5_A d_1 <bass> p_6/2_F d_1/4 p_r d_1/4 p_6/4_F d_1/2 <bar> <melody> p_10/5_A d_1/4 ' \
                   'p_r d_1/4 p_11/5_B d_1/2 <bass> p_1/4_C d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/5_B ' \
                   'd_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1 ' \
                   '<bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1/4 p_r d_1/4 p_11/3_B d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_11/5_B d_1/4 p_r d_1/4 p_2/6_C d_1/2 <bass> p_11/3_B d_1/4 p_r d_1/4 p_7/2_F d_1/2 ' \
                   '<bar> <melody> p_2/6_C d_1 <bass> p_7/2_F d_1 <bar> <melody> p_2/6_C d_1/2 p_11/5_B d_1/2 <bass> ' \
                   'p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> p_7/2_F d_1/2 p_2/4_C ' \
                   'd_1/2 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_10/5_A d_1/4 p_r d_1/4 <bass> p_11/3_B d_1/4 p_r ' \
                   'd_1/4 p_2/3_C d_1/2 <bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_1 ' \
                   '<bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_3/4 p_6/4_F d_1/4 <bass> p_2/3_C d_1/4 p_r d_1/4 ' \
                   'p_9/3_G d_1/4 p_r d_1/4 <bar> <melody> p_4/5_E d_1/4 p_r d_1/4 p_10/5_A d_1/2 <bass> p_9/3_G ' \
                   'd_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> ' \
                   'p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_3/4 p_r d_1/4 <bar> ' \
                   '<melody> p_10/5_A d_1/4 p_r d_1/4 p_11/5_B d_1/2 <bass> p_1/4_C d_1/4 p_r d_1/4 p_11/2_B d_1/2 ' \
                   '<bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1 ' \
                   '<bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_2/6_C ' \
                   'd_1/2 <bass> p_11/2_B d_1/2 p_7/2_F d_1/2 <bar> <melody> p_2/6_C d_1 <bass> p_7/2_F d_1 <bar> ' \
                   '<melody> p_2/6_C d_1/2 p_11/5_B d_1/2 <bass> p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/2 p_1/6_C ' \
                   'd_1/4 p_r d_1/4 <bass> p_7/2_F d_3/4 p_r d_1/4 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_9/5_G ' \
                   'd_1/2 <bass> p_11/3_B d_1/4 p_r d_1/4 p_2/3_C d_1/2 <bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 ' \
                   '<bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_3/4 p_r d_1/4 <bass> ' \
                   'p_2/3_C d_3/4 p_r d_1/4 <bar> <melody> p_4/5_E d_1/4 p_r d_1/4 p_10/5_A d_1/2 <bass> p_r d_1/2 ' \
                   'p_6/2_F d_1/2 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> ' \
                   'p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1/4 p_r d_1/4 p_6/4_F d_1/2 <bar> ' \
                   '<melody> p_10/5_A d_1/4 p_r d_1/4 p_11/5_B d_1/2 <bass> p_1/4_C d_1/4 p_r d_1/4 p_11/2_B d_1/2 ' \
                   '<bar> <melody> p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B ' \
                   'd_1 <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1/4 p_r d_1/4 p_11/3_B ' \
                   'd_1/4 p_r d_1/4 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_2/6_C d_1/2 <bass> p_11/3_B d_1/4 p_r ' \
                   'd_1/4 p_7/2_F d_1/2 <bar> <melody> p_2/6_C d_1 <bass> p_7/2_F d_1 <bar> <melody> p_2/6_C d_1/2 ' \
                   'p_11/5_B d_1/2 <bass> p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/2 p_1/6_C d_1/4 p_r d_1/4 <bass> ' \
                   'p_7/2_F d_1/2 p_2/4_C d_1/2 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_9/5_G d_1/2 <bass> p_11/3_B ' \
                   'd_1/4 p_r d_1/4 p_2/3_C d_1/2 <bar> <melody> p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> ' \
                   'p_9/5_G d_1 <bass> p_2/3_C d_1 <bar> <melody> p_9/5_G d_3/4 p_6/4_F d_1/4 <bass> p_2/3_C d_1/4 ' \
                   'p_r d_1/4 p_9/3_G d_1/4 p_r d_1/4 <bar> <melody> p_4/5_E d_1/4 p_r d_1/4 p_10/5_A d_1/2 <bass> ' \
                   'p_9/3_G d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_10/5_A d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1 <bass> p_6/2_F d_3/4 p_r ' \
                   'd_1/4 <bar> <melody> p_10/5_A d_1/4 p_r d_1/4 p_11/5_B d_1/2 <bass> p_1/4_C d_1/4 p_r d_1/4 ' \
                   'p_11/2_B d_1/2 <bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1/4 p_11/2_B d_1/4 p_11/2_B d_1/2 ' \
                   '<bar> <melody> p_11/5_B d_1 <bass> p_11/2_B d_1/2 p_11/2_B d_1/2 <bar> <melody> p_11/5_B d_1 ' \
                   '<bass> p_11/2_B d_1/4 p_11/2_B d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 ' \
                   'p_11/4_B d_1/2 <bass> <tup> p_11/2_B p_11/2_B p_11/2_B d_1/2 </tup> p_11/2_B d_1/4 p_r d_1/4 ' \
                   '<bar> <melody> p_11/4_B d_1 <bass> p_2/4_C d_1 <bar> <melody> p_11/5_B d_1/4 p_2/6_C d_1/4 ' \
                   'p_6/6_F d_1/4 p_r d_1/4 <bass> p_2/4_C d_1 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 p_11/6_B d_1/4 ' \
                   'p_r d_1/4 <bass> p_2/4_C d_1/2 p_4/4_E d_1/2 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 p_1/6_C d_1/4 ' \
                   'p_r d_1/4 <bass> <tup> p_r p_10/3_A p_6/2_F d_1 </tup> <bar> <melody> p_1/4_C d_1 <bass> p_6/2_F ' \
                   'd_1 <bar> <melody> p_10/5_A d_1/4 p_1/6_C d_1/4 p_4/6_E d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> ' \
                   '<melody> p_1/6_C d_1/4 p_r d_1/4 p_10/6_A d_1/4 p_r d_1/4 <bass> p_6/2_F d_1/2 p_6/2_F d_1/4 p_r ' \
                   'd_1/4 <bar> <melody> p_1/6_C d_1/4 p_7/3_F d_1/4 p_7/4_F d_1/8 p_7/4_F d_3/8 <bass> p_r d_1/2 ' \
                   'p_7/2_F d_1/2 <bar> <melody> p_7/4_F d_1 <bass> p_7/2_F d_1 <bar> <melody> p_7/5_F d_1/4 p_11/5_B ' \
                   'd_1/4 p_2/6_C d_1/4 p_r d_1/4 <bass> p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_7/6_F ' \
                   'd_1/4 p_r d_1/4 <bass> p_7/2_F d_3/4 p_1/4_C d_1/4 <bar> <melody> p_11/5_B d_1/4 p_6/3_F d_1/4 ' \
                   'p_6/4_F d_1/2 <bass> p_r d_1/4 p_10/3_A d_1/4 p_6/2_F d_1/4 p_6/2_F d_1/4 <bar> <melody> p_6/4_F ' \
                   'd_1/2 p_4/6_E d_1/2 <bass> p_6/2_F d_1 <bar> <melody> p_4/6_E d_1/4 p_r d_1/4 p_2/6_C d_1/2 ' \
                   '<bass> p_6/2_F d_1 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 p_1/6_C d_1/2 <bass> p_6/2_F d_1/2 ' \
                   'p_4/4_E d_1/2 <bar> <melody> p_1/6_C d_1/2 p_6/4_F d_1/8 p_11/4_B d_1/8 p_11/4_B d_1/4 <bass> ' \
                   'p_4/4_E d_1/2 p_11/2_B d_1/2 <bar> <melody> <tup> p_11/4_B p_11/4_B p_6/3_F d_1 </tup> <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_11/5_B d_1/4 p_2/6_C d_1/4 p_6/6_F d_1/4 p_r d_1/4 <bass> p_11/2_B ' \
                   'd_1 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 p_11/6_B d_1/4 p_r d_1/4 <bass> p_11/2_B d_1/2 p_r ' \
                   'd_1/2 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 p_1/6_C d_1/4 p_r d_1/4 <bass> <tup> p_r p_6/3_F ' \
                   'p_6/2_F d_1 </tup> <bar> <melody> p_7/4_F d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1/4 ' \
                   'p_1/6_C d_1/4 p_4/6_E d_1/4 p_6/2_F d_1/4 <bass> p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> p_1/6_C ' \
                   'd_1/4 p_r d_1/4 p_10/6_A d_1/4 p_r d_1/4 <bass> p_6/2_F d_1/2 p_6/2_F d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_1/6_C d_1/4 p_7/3_F d_1/4 p_7/4_F d_1/2 <bass> p_r d_1/2 p_7/2_F d_1/2 <bar> <melody> ' \
                   'p_7/4_F d_1 <bass> p_7/2_F d_1 <bar> <melody> p_7/5_F d_1/4 p_11/5_B d_1/4 p_2/6_C d_1/4 p_r ' \
                   'd_1/4 <bass> p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_7/6_F d_1/4 p_r d_1/4 <bass> ' \
                   'p_7/2_F d_1/4 p_r d_1/4 p_11/3_B d_1/4 p_r d_1/4 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_6/6_F ' \
                   'd_1/2 <bass> p_6/4_F d_1/2 p_6/2_F d_1/2 <bar> <melody> p_5/6_E d_1/4 p_r d_1/4 p_6/6_F d_1/2 ' \
                   '<bass> p_6/2_F d_1 <bar> <melody> p_5/6_E d_1/4 p_r d_1/4 p_6/6_F d_1/2 <bass> p_6/2_F d_1/2 ' \
                   'p_6/2_F d_1/2 <bar> <melody> p_5/6_E d_1/4 p_r d_1/4 p_6/6_F d_1/2 <bass> p_6/2_F d_1/2 p_6/2_F ' \
                   'd_1/2 <bar> <melody> p_5/6_E d_1/4 p_r d_1/4 p_6/5_F d_1/4 p_r d_1/4 <bass> <tup> p_10/3_A ' \
                   'p_11/3_B p_11/2_B d_1 </tup> <bar> <melody> p_11/4_B d_1/4 p_r d_1/4 p_11/4_B d_1/2 <bass> ' \
                   'p_11/2_B d_3/4 p_11/2_B d_1/4 <bar> <melody> p_11/4_B d_3/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> ' \
                   '<melody> p_11/4_B d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 <bass> p_11/2_B d_1/4 p_r d_1/4 ' \
                   'p_11/3_B d_1/4 p_r d_1/4 <bar> <melody> p_2/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 <bass> ' \
                   'p_r d_1/2 p_6/2_F d_1/2 <bar> <melody> p_1/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 <bass> ' \
                   'p_6/2_F d_1 <bar> <melody> p_12/4_B d_1/4 p_r d_1/4 p_1/5_C d_1/2 <bass> p_6/2_F d_1/2 p_6/2_F ' \
                   'd_1/2 <bar> <melody> p_6/4_F d_1/4 p_6/2_F d_1/4 p_1/4_C d_1/4 p_r d_1/4 <bass> p_6/2_F d_1/4 p_r ' \
                   'd_1/4 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_11/4_B d_1/4 p_r d_1/4 p_1/5_C d_1/4 p_r d_1/4 ' \
                   '<bass> p_r d_1/2 p_4/2_E d_1/2 <bar> <melody> p_7/4_F d_1/4 p_r d_1/4 p_7/4_F d_1/2 <bass> ' \
                   'p_4/2_E d_1 <bar> <melody> p_7/4_F d_1/2 p_4/4_E d_1/2 <bass> p_4/2_E d_1 <bar> <melody> p_7/4_F ' \
                   'd_1/4 p_r d_1/4 p_9/4_G d_1/4 p_r d_1/4 <bass> p_4/2_E d_1/4 p_r d_3/4 <bar> <melody> p_11/4_B ' \
                   'd_1/4 p_r d_1/4 p_2/5_C d_1/4 p_r d_1/4 <bass> p_r d_1/2 p_11/2_B d_1/2 <bar> <melody> p_2/5_C ' \
                   'd_1/4 p_r d_1/4 p_2/5_C d_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> <melody> p_1/5_C d_1/4 p_r ' \
                   'd_1/4 p_2/5_C d_1/4 p_r d_1/4 <bass> p_r d_1/4 p_11/3_B d_1/4 p_11/3_B d_1/4 p_r d_1/4 <bar> ' \
                   '<melody> p_6/5_F d_1 <bass> p_11/2_B d_1 <bar> <melody> p_4/5_E d_1/4 p_11/4_B d_1/4 p_6/5_F ' \
                   'd_1/4 p_11/3_B d_1/4 <bass> p_11/2_B d_1/4 p_r d_1/4 p_11/2_B d_1/2 <bar> <melody> p_11/4_B d_1/4 ' \
                   'p_r d_1/4 p_11/4_B d_1/2 <bass> p_11/2_B d_1 <bar> <melody> p_11/4_B d_3/4 p_r d_1/4 <bass> ' \
                   'p_11/2_B d_1 <bar> <melody> p_11/4_B d_1/4 p_r d_1/8 p_2/4_C d_1/8 p_1/5_C d_1/4 p_r d_1/4 <bass> ' \
                   'p_11/2_B d_1/2 p_11/3_B d_1/8 p_r d_3/8 <bar> <melody> p_2/5_C d_1/4 p_r d_1/4 p_6/5_F d_1/4 ' \
                   'p_9/3_G d_1/4 <bass> p_r d_1/2 p_6/2_F d_1/2 <bar> <melody> p_1/5_C d_1/4 p_r d_1/4 p_1/5_C d_1/4 ' \
                   'p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_2/5_C d_1/4 <tup> p_r p_9/3_G p_6/4_F d_1/4 </tup> ' \
                   'p_1/5_C d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_11/4_B d_1/4 p_r d_1/8 p_1/4_C d_1/8 ' \
                   'p_1/5_C d_1/4 p_r d_1/4 <bass> p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> p_11/4_B d_1/4 p_r d_1/4 ' \
                   'p_2/5_C d_1/2 <bass> <tup> p_r p_7/3_F p_4/2_E d_1 </tup> <bar> <melody> p_2/5_C d_1/2 p_7/4_F ' \
                   'd_1/4 p_r d_1/4 <bass> p_4/2_E d_1 <bar> <melody> p_7/4_F d_1/2 p_4/4_E d_1/4 p_r d_1/4 <bass> ' \
                   'p_4/2_E d_3/8 p_4/2_E d_1/8 p_4/2_E d_1/2 <bar> <melody> p_7/4_F d_1/4 p_r d_1/4 p_9/4_G d_1/4 ' \
                   'p_r d_1/4 <bass> p_4/2_E d_1/4 p_r d_3/4 <bar> <melody> p_11/4_B d_1/4 p_11/3_B d_1/4 p_4/5_E ' \
                   'd_1/2 <bass> p_r d_1/2 p_11/2_B d_1/2 <bar> <melody> p_2/5_C d_1 <bass> p_11/2_B d_1 <bar> ' \
                   '<melody> p_1/5_C d_1/4 p_r d_1/4 p_2/5_C d_1/2 <bass> p_11/2_B d_1 <bar> <melody> p_11/4_B d_1/4 ' \
                   'p_r d_1/8 p_11/3_B d_1/8 p_11/4_B d_1/2 <bass> p_11/2_B d_3/4 p_r d_1/8 p_11/3_B d_1/8 <bar> ' \
                   '<melody> p_11/4_B d_1 <bass> p_r d_1/2 p_11/2_B d_1/2 <bar> <melody> p_11/4_B d_1 <bass> p_11/2_B ' \
                   'd_1 <bar> <melody> p_11/5_B d_1/4 p_2/6_C d_1/4 p_6/6_F d_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> ' \
                   '<melody> p_2/6_C d_1/4 p_r d_1/4 p_11/6_B d_1/4 p_r d_1/4 <bass> p_11/2_B d_1/2 p_4/4_E d_1/2 ' \
                   '<bar> <melody> p_2/6_C d_1/4 p_r d_1/4 p_1/6_C d_1/4 p_r d_1/4 <bass> <tup> p_r p_10/3_A p_6/2_F ' \
                   'd_1 </tup> <bar> <melody> p_1/4_C d_1 <bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1/4 p_1/6_C ' \
                   'd_1/4 p_4/6_E d_1/4 p_r d_1/4 <bass> p_6/2_F d_1 <bar> <melody> p_1/6_C d_1/4 p_r d_1/4 p_10/6_A ' \
                   'd_1/4 p_r d_1/4 <bass> p_6/2_F d_1/2 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_1/6_C d_1/4 p_7/3_F ' \
                   'd_1/4 p_7/4_F d_1/8 p_7/4_F d_3/8 <bass> p_r d_1/2 p_7/2_F d_1/2 <bar> <melody> p_7/4_F d_1 ' \
                   '<bass> p_7/2_F d_1 <bar> <melody> p_7/5_F d_1/4 p_11/5_B d_1/4 p_2/6_C d_1/4 p_r d_1/4 <bass> ' \
                   'p_7/2_F d_1 <bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_7/6_F d_1/4 p_r d_1/4 <bass> p_7/2_F d_3/4 ' \
                   'p_1/4_C d_1/4 <bar> <melody> p_11/5_B d_1/4 p_6/3_F d_1/4 p_6/4_F d_1/2 <bass> p_r d_1/4 p_10/3_A ' \
                   'd_1/4 p_6/2_F d_1/4 p_6/2_F d_1/4 <bar> <melody> p_6/4_F d_1/2 p_4/6_E d_1/2 <bass> p_6/2_F d_1 ' \
                   '<bar> <melody> p_4/6_E d_1/4 p_r d_1/4 p_2/6_C d_1/2 <bass> p_6/2_F d_1 <bar> <melody> p_2/6_C ' \
                   'd_1/4 p_r d_1/4 p_1/6_C d_1/2 <bass> p_6/2_F d_1/2 p_4/4_E d_1/2 <bar> <melody> p_1/6_C d_1/2 ' \
                   'p_6/4_F d_1/8 p_11/4_B d_1/8 p_11/4_B d_1/4 <bass> p_4/4_E d_1/2 p_11/2_B d_1/2 <bar> <melody> ' \
                   '<tup> p_11/4_B p_11/4_B p_6/3_F d_1 </tup> <bass> p_11/2_B d_1 <bar> <melody> p_11/5_B d_1/4 ' \
                   'p_2/6_C d_1/4 p_6/6_F d_1/4 p_r d_1/4 <bass> p_11/2_B d_1 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 ' \
                   'p_11/6_B d_1/4 p_r d_1/4 <bass> p_11/2_B d_1/2 p_r d_1/2 <bar> <melody> p_2/6_C d_1/4 p_r d_1/4 ' \
                   'p_1/6_C d_1/4 p_r d_1/4 <bass> <tup> p_r p_6/3_F p_6/2_F d_1 </tup> <bar> <melody> p_7/4_F d_1 ' \
                   '<bass> p_6/2_F d_1 <bar> <melody> p_10/5_A d_1/4 p_1/6_C d_1/4 p_4/6_E d_1/4 p_6/2_F d_1/4 <bass> ' \
                   'p_6/2_F d_3/4 p_r d_1/4 <bar> <melody> p_1/6_C d_1/4 p_r d_1/4 p_10/6_A d_1/4 p_r d_1/4 <bass> ' \
                   'p_6/2_F d_1/2 p_6/2_F d_1/4 p_r d_1/4 <bar> <melody> p_1/6_C d_1/4 p_7/3_F d_1/4 p_7/4_F d_1/2 ' \
                   '<bass> p_r d_1/2 p_7/2_F d_1/2 <bar> <melody> p_7/4_F d_1 <bass> p_7/2_F d_1 <bar> <melody> ' \
                   'p_7/5_F d_1/4 p_11/5_B d_1/4 p_2/6_C d_1/4 p_r d_1/4 <bass> p_7/2_F d_1 <bar> <melody> p_11/5_B ' \
                   'd_1/4 p_r d_1/4 p_7/6_F d_1/4 p_r d_1/4 <bass> p_7/2_F d_1/4 p_r d_1/4 p_11/3_B d_1/4 p_r d_1/4 ' \
                   '<bar> <melody> p_11/5_B d_1/4 p_r d_1/4 p_6/6_F d_1/2 <bass> p_6/4_F d_1/2 p_6/2_F d_1/2 <bar> ' \
                   '<melody> p_5/6_E d_1/4 p_r d_1/4 p_6/6_F d_1/2 <bass> p_6/2_F d_1 <bar> <melody> p_5/6_E d_1/4 ' \
                   'p_r d_1/4 p_6/6_F d_1/2 <bass> p_6/2_F d_1/2 p_6/2_F d_1/2 <bar> <melody> p_5/6_E d_1/4 p_r d_1/4 ' \
                   'p_6/6_F d_1/2 <bass> p_6/2_F d_1/2 p_6/2_F d_1/2 <bar> <melody> p_5/6_E d_1/4 p_6/2_F d_1/4 ' \
                   'p_6/6_F d_1/2 <bass> p_6/2_F d_1/4 p_r d_1/4 p_6/2_F d_1/2 <bar> <melody> p_6/6_F d_1 <bass> ' \
                   'p_6/2_F d_1 <bar> <melody> p_6/6_F d_1 <bass> p_6/2_F d_1 <bar> <melody> p_6/6_F d_1 <bass> ' \
                   'p_6/2_F d_1 <bar> <melody> p_6/6_F d_1 <bass> p_6/2_F d_1 <bar> <melody> p_6/6_F d_1/2 p_r d_1/2 ' \
                   '<bass> p_6/2_F d_1/2 p_r d_1/2 </s>'

gen_broken = 'TimeSig_4/4 Tempo_112 Key_BbMinor <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bass> ' \
              'p_7/5_5 d_6 <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_9/6_6 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_9/6_6 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bass> p_7/5_5 d_6 <bar> ' \
              '<melody> p_4/6_4 d_1/2 p_2/6_2 d_1/2 p_4/6_4 d_1/2 p_9/6_6 d_1/2 p_4/6_4 d_1/2 p_2/6_2 d_1/2 p_4/6_4 d_1/2 p_2/6_2 ' \
              'd_1/2 p_4/6_4 d_1/2 p_9/6_6 d_1/2 p_4/6_4 d_1/2 p_2/6_2 d_1/2 <bass> p_6/5_5 d_1/8 p_6/5_5 d_3/8 p_6/5_5 d_1 p_6/5_5 ' \
              'd_3 p_6/5_5 d_1/2 p_r d_1/2 p_2/6_2 d_1/4 p_11/4_1 d_1/4 <bar> <melody> p_4/6_4 d_1/2 p_2/6_2 d_1/2 p_4/6_4 d_1/2 ' \
              'p_9/6_6 d_1/2 p_4/6_4 d_1/2 p_2/6_2 d_1/2 p_1/6_2 d_1/2 p_11/5_1 d_1/2 p_1/6_2 d_1/2 p_6/6_5 d_1/2 p_1/6_2 d_1/2 ' \
              'p_11/5_1 d_1/2 <bass> p_11/4_1 d_3/2 p_r d_3/2 p_9/4_6 d_3 <bar> <melody> p_6/6_5 d_1/2 <bass> p_11/4_1 d_6 <melody> ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> ' \
              'p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> ' \
              'p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 ' \
              '<melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              '<bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> ' \
              'p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 ' \
              '<bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bass> ' \
              'p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> ' \
              '<melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 ' \
              '<melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 ' \
              '<melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> ' \
              'p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 ' \
              '<melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> ' \
              'p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 ' \
              '<melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 ' \
              '<melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> ' \
              '<melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 ' \
              '<melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 ' \
              '<melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 ' \
              '<melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 ' \
              'd_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> ' \
              'p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> ' \
              'p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> ' \
              'p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 ' \
              '<melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> ' \
              '<bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 ' \
              '<bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 ' \
              'd_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 ' \
              'p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 <bar> <melody> p_6/6_5 d_1/2 <bass> p_6/5_5 d_6 <melody> p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 ' \
              'd_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 ' \
              'd_1/2 <bar> <bass> p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 ' \
              'p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 <bar> <bass> ' \
              'p_6/5_5 d_6 <melody> p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_3/6_3 d_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2 p_6/6_5 ' \
              'd_1/2 p_4/6_4 d_1/2 p_6/6_5 d_1/2 p_11/6_1 d_1/2 p_6/6_5 d_1/2 p_4/6_4 d_1/2'
