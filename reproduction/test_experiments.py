import unittest
import numpy as np
from experiments import (LinUCBDisjoint,LinUCBHybrid,ContextualGreedy,EpsilonGreedy,
                         replay,features,metrics,context,make_policy)

class CorrectnessTests(unittest.TestCase):
    def test_click_direction_and_no_click(self):
        p=LinUCBDisjoint(2,2);x=np.array([1.,0.]);p.update(0,x,1)
        self.assertGreater(p.theta[0,0],0)
        before=p.b.copy();p.update(1,x,0);np.testing.assert_array_equal(p.b,before)
    def test_rank_one_matches_exact_inverse(self):
        p=LinUCBDisjoint(1,3);rng=np.random.default_rng(7);A=np.eye(3);b=np.zeros(3)
        for _ in range(50):
            x=rng.normal(size=3);r=int(rng.random()<.3);p.update(0,x,r);A+=np.outer(x,x);b+=r*x
        np.testing.assert_allclose(p.inv[0],np.linalg.inv(A),atol=1e-11)
        np.testing.assert_allclose(p.theta[0],np.linalg.solve(A,b),atol=1e-11)
    def test_window_expires_during_unmatched_events(self):
        p=LinUCBDisjoint(1,2,window=1);x=np.array([[1.,0.]])
        p.choose([0],x);p.update(0,x[0],1);p.choose([0],x);p.choose([0],x)
        np.testing.assert_allclose(p.inv[0],np.eye(2));np.testing.assert_allclose(p.b,0)
    def test_unmatched_reward_never_used(self):
        class Fixed:
            def choose(self,arms,x):return 0
            def update(self,*args):raise AssertionError('Unmatched feedback used')
        row=(0,np.zeros(6),1,1,np.array([0,1]))
        z=replay([row],np.zeros((2,6)),Fixed())
        self.assertEqual(z['matched'],0);self.assertEqual(z['clicks'],0)
    def test_features_have_no_label(self):
        x=features(np.ones(6),np.ones((2,6)))
        self.assertEqual(x.shape,(2,48));np.testing.assert_array_equal(x,1)
    def test_hybrid_context_is_36d_outer_product(self):
        # Li et al. (2010), Sec. 5.2.2: z_{t,a} = user (x) article, flattened to 36.
        af=np.arange(3*6,dtype=float).reshape(3,6)+1
        u=np.array([1.,2.,3.,4.,5.,6.]);arms=np.array([2,0])
        x,z=context(u,arms,af,hybrid=True)
        self.assertEqual(x.shape,(2,6));self.assertEqual(z.shape,(2,36))
        for row,a in enumerate(arms):
            np.testing.assert_allclose(z[row],np.outer(u,af[a]).ravel())
        p=LinUCBHybrid(3)  # the production default must already be 36-dimensional
        self.assertEqual(p.k,36);self.assertEqual(p.A0.shape,(36,36))
        p.update(0,u,z[0],1)
        self.assertEqual(p.beta.shape,(36,));self.assertEqual(p.B.shape,(3,6,36))
    def test_hybrid_matches_joint_ridge_solution(self):
        p=LinUCBHybrid(2,d=2,k=3,alpha=.25,seed=5)
        rng=np.random.default_rng(11)
        data=[]
        for _ in range(40):
            a=int(rng.integers(2));x=rng.normal(size=2);z=rng.normal(size=3)
            r=int(rng.random()<.4);p.update(a,x,z,r);data.append((a,x,z,r))
        A0=np.eye(3)+sum(np.outer(v[2],v[2]) for v in data)
        b0=sum(v[3]*v[2] for v in data)
        per_arm=[]
        for a in range(2):
            rows=[v for v in data if v[0]==a]
            X=np.array([v[1] for v in rows]);Z=np.array([v[2] for v in rows]);y=np.array([v[3] for v in rows])
            per_arm.append((np.eye(2)+X.T@X, X.T@Z, X.T@y))
        S=A0-sum(B.T@np.linalg.inv(A)@B for A,B,b in per_arm)
        beta=np.linalg.solve(S,b0-sum(B.T@np.linalg.solve(A,b) for A,B,b in per_arm))
        np.testing.assert_allclose(p.beta,beta,atol=1e-9)
        for a,(A,B,b) in enumerate(per_arm):
            np.testing.assert_allclose(p.theta[a],np.linalg.solve(A,b-B@beta),atol=1e-9)
    def test_hybrid_ucb_variance_matches_algorithm2(self):
        # Li et al. 2010 Algorithm 2, line 13 (full quadratic form).
        p=LinUCBHybrid(3,d=2,k=3,alpha=.5,seed=9)
        rng=np.random.default_rng(23)
        for _ in range(30):
            p.update(int(rng.integers(3)),rng.normal(size=2),rng.normal(size=3),int(rng.random()<.5))
        arms=np.array([1,2]);x=rng.normal(size=(2,2));z=rng.normal(size=(2,3))
        inv=p.inv[arms];B=p.B[arms]
        expected=np.array([z[i]@p.S_inv@z[i]
            -2*z[i]@p.S_inv@B[i].T@inv[i]@x[i]
            +x[i]@inv[i]@x[i]
            +x[i]@inv[i]@B[i]@p.S_inv@B[i].T@inv[i]@x[i] for i in range(2)])
        A_inv_B=np.einsum('ijk,ikl->ijl',inv,B)
        w=np.einsum('ij,ijk->ik',x,A_inv_B)
        got=np.einsum('ij,ijk,ik->i',x,inv,x)+np.einsum('ik,kl,il->i',z-w,p.S_inv,z-w)
        np.testing.assert_allclose(got,expected,atol=1e-10)
    def test_hybrid_choose_runs_over_candidate_set(self):
        p=LinUCBHybrid(4,d=2,k=3,alpha=.5,seed=1)
        rng=np.random.default_rng(3)
        for _ in range(10):
            p.update(int(rng.integers(4)),rng.normal(size=2),rng.normal(size=3),int(rng.random()<.5))
        arms=np.array([0,2,3]);x=rng.normal(size=(3,2));z=rng.normal(size=(3,3))
        a=p.choose(arms,x,z)
        self.assertIn(int(a),list(arms))
    def test_metrics_match_known_confusion(self):
        z=metrics(np.array([0,0,1,1]),np.array([.1,.8,.2,.9]),.5)
        self.assertEqual(z['confusion_matrix'],[[1,1],[1,1]])
        self.assertAlmostEqual(z['f1'],.5)
    def test_contextual_greedy_is_zero_width_disjoint_greedy(self):
        p=ContextualGreedy(3,d=2,epsilon=0.,seed=4)
        self.assertEqual(p.alpha,0.)
        self.assertIsInstance(p,LinUCBDisjoint)
        p.theta[:]=np.array([[.1,.9],[.8,.2],[.3,.7]])
        arms=np.array([0,1,2]);x=np.array([[1.,0.]]*3)
        self.assertEqual(p.choose(arms,x),1)
    def test_epsilon_greedy_ignores_context(self):
        p=EpsilonGreedy(4,epsilon=0.,seed=8)
        arms=np.array([1,2,3])
        self.assertEqual(p.choose(arms,np.zeros((3,6))),p.choose(arms,np.ones((3,6))*99))
    def test_epsilon_greedy_uses_empirical_means(self):
        p=EpsilonGreedy(4,epsilon=0.,seed=8)
        p.update(0,None,1);p.update(1,None,0);p.update(2,None,1)
        np.testing.assert_allclose(p.means[:3],[1.,0.,1.])
        self.assertEqual(p.choose(np.array([0,1,2]),None),0)  # first tie retained
    def test_policy_factory_returns_paper_aligned_classes(self):
        self.assertIsInstance(make_policy('disjoint',.5,4),LinUCBDisjoint)
        self.assertIsInstance(make_policy('hybrid',.5,4),LinUCBHybrid)
        self.assertIsInstance(make_policy('contextual_greedy',.05,4),ContextualGreedy)
        self.assertIsInstance(make_policy('epsilon_greedy',.05,4),EpsilonGreedy)

if __name__=='__main__':unittest.main()
