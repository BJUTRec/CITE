# import tensorflow as tf            ### if running in tf1.x please enable it !

import tensorflow.compat.v1 as tf    ### if running in tf 2.x please enable it
tf.disable_v2_behavior()             ### if running in tf 2.x please enable it

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utility.helper import *
from utility.batch_test import *
from tqdm import tqdm


class CITE(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        # graph data
        self.n_fold = 100
        self.plain_adj = data_config['plain_adj']
        self.norm_adj = data_config['norm_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.n_nonzero_elems = self.plain_adj.count_nonzero()

        # model setting
        self.emb_dim = args.embed_size
        self.n_factors = args.n_factors
        self.n_iterations = args.n_iterations
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads

        self.pick_level = args.pick_scale
        if args.pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False

        # training setting
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        # tf.compat.v1.disable_eager_execution()
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        # create models
        self.ua_embeddings, self.ia_embeddings = self.forward(pick_=self.is_pick)

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        # Inference for the testing phase.
        self.batch_ratings = tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t, transpose_a=False, transpose_b=True)

        # Generate Predictions & Optimize via BPR loss.
        self.mf_loss = self.create_bpr_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings)
        # self.mf_loss = self.create_ssm_loss(self.u_g_embeddings, self.pos_i_g_embeddings)  ### if loss==nan when running other dataset with bpr loss, please enable ssm loss instead of bpr loss!!!

        # self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        self.loss = self.mf_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        # initializer = tf.contrib.layers.xavier_initializer(uniform=False)  ### if running in tf1.x please enable it !
        initializer = tf.keras.initializers.glorot_normal()     ### if running in tf2.x please enable it !

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            for k in range(self.n_factors):
                all_weights['W_q_%d' % k] = tf.Variable(initializer([int(self.emb_dim/self.n_factors), int(self.emb_dim/self.n_factors)]),
                                                        name='W_q_%d' % k)
                all_weights['W_k_%d' % k] = tf.Variable(initializer([int(self.emb_dim/self.n_factors), int(self.emb_dim/self.n_factors)]),
                                                        name='W_k_%d' % k)
                all_weights['W_v_%d' % k] = tf.Variable(initializer([int(self.emb_dim/self.n_factors), int(self.emb_dim/self.n_factors)]),
                                                        name='W_v_%d' % k)
            print('using xavier initialization')

        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True, name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True, name='item_embedding', dtype=tf.float32)

            for k in range(self.n_factors):
                all_weights['W_q_%d' % k] = tf.Variable(
                    initializer([int(self.emb_dim / self.n_factors), int(self.emb_dim / self.n_factors)]),
                    name='W_q_%d' % k)
                all_weights['W_k_%d' % k] = tf.Variable(
                    initializer([int(self.emb_dim / self.n_factors), int(self.emb_dim / self.n_factors)]),
                    name='W_k_%d' % k)
                all_weights['W_v_%d' % k] = tf.Variable(
                    initializer([int(self.emb_dim / self.n_factors), int(self.emb_dim / self.n_factors)]),
                    name='W_v_%d' % k)

            print('using pretrained initialization')
        return all_weights

    def forward(self, pick_=False):
        p_train = False
        p_test = False

        # 生成单通道embeddings
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        # Disentangled GCN
        A_values = tf.ones(shape=[self.n_factors, len(self.all_h_list)])

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            n_factors_l = self.n_factors
            n_iterations_l = self.n_iterations
            layer_embeddings = []

            # 将多通道embeddings分割 every split size: [(n_users+u_items),embed_size]
            ego_layer_embeddings = tf.split(ego_embeddings, n_factors_l, 1)

            # 进行路由更新机制
            for t in range(0, n_iterations_l):
                iter_embeddings = []
                A_iter_values = []

                if t == n_iterations_l:
                    p_test = pick_
                    p_train = False

                # 产生更新后度矩阵和邻接矩阵list  block nums in list: n_factors       size:[n_users+n_items,n_users+n_items]
                A_factors, D_col_factors, D_row_factors = self._updated_graph(n_factors_l, A_values, pick=p_train)

                for i in range(0, n_factors_l):
                    # simplified GCN to update embeddings
                    factors_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], ego_layer_embeddings[i])
                    factors_embeddings = tf.sparse.sparse_dense_matmul(A_factors[i], factors_embeddings)
                    factors_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], factors_embeddings)
                    iter_embeddings.append(factors_embeddings)

                    if t == n_iterations_l - 1:
                        layer_embeddings = iter_embeddings

                    # 头embeddings = 更新后u i embeddings
                    head_factor_embeddings = tf.nn.embedding_lookup(factors_embeddings, self.all_h_list)
                    # 尾embeddings = 迭代前初始 u0 i0 embeddings
                    tail_factor_embeddings = tf.nn.embedding_lookup(ego_layer_embeddings[i], self.all_t_list)

                    # 归一化 缩小到（0，1）
                    head_factor_embeddings = tf.math.l2_normalize(head_factor_embeddings, axis=1)
                    tail_factor_embeddings = tf.math.l2_normalize(tail_factor_embeddings, axis=1)

                    # 各通道内邻接矩阵各交互元素更新（亲和力计算）size:[all_h_list,1]
                    A_factors_values = tf.reduce_sum(tf.multiply(head_factor_embeddings, tf.tanh(tail_factor_embeddings)), axis=1)
                    A_iter_values.append(A_factors_values)

                # 将更新后各通道A list转化为n_factors A list   size:[n_factors,all_h_list]
                A_iter_values = tf.stack(A_iter_values, 0)
                A_values += A_iter_values

            # 各层n通道embeddings拼接 size：[n_users+n_items,embed_size * n_factors * n_layers]
            side_embeddings = tf.concat(layer_embeddings, 1)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        ui_factor_embeddings = tf.split(all_embeddings, self.n_factors, 1)

        #Transformer block - single layer
        all_embeddings = self.selfAttention(ui_factor_embeddings, number=self.n_factors, inpDim=int(self.emb_dim/self.n_factors))
        all_embeddings = tf.concat(all_embeddings, axis=1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def selfAttention(self, localReps, number, inpDim):
        attReps = [None] * number
        stkReps = tf.stack(localReps, axis=1)
        for i in range(number):
            glbRep = localReps[i]
            temAttRep = self.multiHeadAttention(stkReps, glbRep, number=number, numHeads=self.n_heads, inpDim=inpDim, k=i) + glbRep
            attReps[i] = temAttRep
        return attReps

    def multiHeadAttention(self, localReps, glbRep, number, numHeads, inpDim, k):
        query = tf.reshape(tf.tile(tf.reshape(tf.matmul(glbRep, self.weights['W_q_%d' % k]), [-1, 1, inpDim]), [1, number, 1]), [-1, numHeads, inpDim//numHeads])
        temLocals = tf.reshape(localReps, [-1, inpDim])
        key = tf.reshape(tf.matmul(temLocals, self.weights['W_k_%d' % k]), [-1, numHeads, inpDim//numHeads])
        val = tf.reshape(tf.matmul(temLocals, self.weights['W_v_%d' % k]), [-1, number, numHeads, inpDim//numHeads])
        att = tf.nn.softmax(2*tf.reshape(tf.reduce_sum(query * key, axis=-1), [-1, number, numHeads, 1]), axis=1)
        attRep = tf.reshape(tf.reduce_sum(val * att, axis=1), [-1, inpDim])
        return attRep

    def _updated_graph(self, f_num, A_factor_values, pick=True):
        A_factors = []
        D_col_factors = []
        D_row_factors = []
        # get the indices of adjacency matrix.
        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        D_indices = np.mat(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))]).transpose()

        # apply factor-aware softmax function over the values of adjacency matrix
        # .... A_factor_values is [n_factors, all_h_list]
        if pick:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)
            min_A = tf.reduce_min(A_factor_scores, 0)
            index = A_factor_scores > (min_A + 0.0000001)
            index = tf.cast(index, tf.float32) * (
                        self.pick_level - 1.0) + 1.0  # adjust the weight of the minimum factor to 1/self.pick_level

            A_factor_scores = A_factor_scores * index
            A_factor_scores = A_factor_scores / tf.reduce_sum(A_factor_scores, 0)
        else:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)

        for i in range(0, f_num):
            # in the i-th factor, couple the adjacency values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            A_i_scores = A_factor_scores[i]
            A_i_tensor = tf.SparseTensor(A_indices, A_i_scores, self.A_in_shape)

            # get the degree values of A_i_tensor
            # .... D_i_scores_col is [n_users+n_items, 1]
            # .... D_i_scores_row is [1, n_users+n_items]
            D_i_col_scores = 1 / tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=1))
            D_i_row_scores = 1 / tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=0))

            # couple the laplacian values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            D_i_col_tensor = tf.SparseTensor(D_indices, D_i_col_scores, self.A_in_shape)
            D_i_row_tensor = tf.SparseTensor(D_indices, D_i_row_scores, self.A_in_shape)

            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)

        # return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        #         maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        #         mf_loss = tf.negative(tf.reduce_mean(maxi))

        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.

        # regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)
        # regularizer = regularizer / self.batch_size
        # emb_loss = self.decay * regularizer

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        return mf_loss

    def create_ssm_loss(self, users, pos_items):
        self.ssm_temp = tf.constant(0.2)
        norm_users = tf.nn.l2_normalize(users, axis=1)
        norm_pos_items = tf.nn.l2_normalize(pos_items, axis=1)

        pos_score = tf.reduce_sum(tf.multiply(norm_users, norm_pos_items), axis=1)
        ttl_score = tf.matmul(norm_users, norm_pos_items, transpose_a=False, transpose_b=True)

        pos_score = tf.exp(pos_score / self.ssm_temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.ssm_temp), axis=1)

        loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))

        # regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)
        # regularizer = regularizer / self.batch_size
        # emb_loss = self.decay * regularizer

        return loss

    def model_save(self, ses):
        save_pretrain_path = '../output_parameters/ml1m'
        np.savez(save_pretrain_path, user_embed=np.array(self.weights['user_embedding'].eval(session=ses)),
                 item_embed=np.array(self.weights['item_embedding'].eval(session=ses)),
                 W_q_0=np.array(self.weights['W_q_0'].eval(session=ses)),
                 W_k_0=np.array(self.weights['W_k_0'].eval(session=ses)),
                 W_v_0=np.array(self.weights['W_v_0'].eval(session=ses)),
                 W_q_1=np.array(self.weights['W_q_1'].eval(session=ses)),
                 W_k_1=np.array(self.weights['W_k_1'].eval(session=ses)),
                 W_v_1=np.array(self.weights['W_v_1'].eval(session=ses)),
                 W_q_2=np.array(self.weights['W_q_2'].eval(session=ses)),
                 W_k_2=np.array(self.weights['W_k_2'].eval(session=ses)),
                 W_v_2=np.array(self.weights['W_v_2'].eval(session=ses)),
                 W_q_3=np.array(self.weights['W_q_3'].eval(session=ses)),
                 W_k_3=np.array(self.weights['W_k_3'].eval(session=ses)),
                 W_v_3=np.array(self.weights['W_v_3'].eval(session=ses)))

def load_best(name="best_model"):
    pretrain_path = "../pretrain_parameters/ml1m_emb.npz"
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the best model:', name)
    except Exception:
        pretrain_data = None
    return pretrain_data

def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list


if __name__ == '__main__':
    whether_test_batch = True

    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    print("************************************************************************************")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)

    config['plain_adj'] = plain_adj
    config['norm_adj'] = pre_adj
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list

    t0 = time()
    """
    *********************************************************
    pretrain = 1: load embeddings with name such as embedding_xxx(.npz), l2_best_model(.npz)
    pretrain = 0: default value, no pretrained embeddings.
    """
    if args.pretrain == 1:
        print("Try to load pretain: ", args.embed_name)
        pretrain_data = load_best(name=args.embed_name)
        if pretrain_data == None:
            print("Load pretrained model(%s)fail!!!!!!!!!!!!!!!" % (args.embed_name))
    else:
        pretrain_data = None

    model = CITE(data_config=config, pretrain_data=pretrain_data)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss = 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = data_generator.sample()

            _, batch_loss, batch_mf_loss = sess.run([model.opt, model.loss, model.mf_loss],
                                                                                    feed_dict={model.users: users,
                                                                                               model.pos_items: pos_items,
                                                                                               model.neg_items: neg_items,
                                                                                               })
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            print(mf_loss)
            sys.exit()

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f ]' % (
            epoch, time() - t1, loss, mf_loss)
            print(perf_str)
        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.show_step != 0:
            #if args.verbose > 0 and epoch % args.verbose == 0:
            # Skip testing
            continue

        # Begin test at this epoch.
        loss_test, mf_loss_test = 0., 0.
        for idx in tqdm(range(n_batch)):

            users, pos_items, neg_items = data_generator.sample_test()
            batch_loss_test, batch_mf_loss_test = sess.run(
                [model.loss, model.mf_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items,
                           })
            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch


        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True, batch_test_flag=whether_test_batch)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f ], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (
                       epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test,  ret['recall'][0],
                       ret['recall'][-1],
                       ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                       ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step,
                                                                    expected_order='acc', flag_step=args.early)

        # early stopping when cur_best_pre_0 is decreasing for given steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            model.model_save(sess)
            print('save the model with performance: ', cur_best_pre_0)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)