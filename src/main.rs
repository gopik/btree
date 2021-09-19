use std::time::Instant;
use test_env_log::test;

/// In memory B-Tree implementation
///
/// * Reference: <http://opendatastructures.org/>
///
/// ## Btree structure
/// The nodes of btree are defined by a branching factor with a maximum of `2B - 1`
/// keys. Each branch node except root has the invariant that number of keys is
/// between `B - 1` and `2B -1` and number of children is 1 greater than the number of keys.
///
/// The leaf nodes have maximum of `2B - 1` keys and values and minimum of `B -1` keys
/// and values. The root must have at least 1 key and 2 children and a maximum of `2B -1`
/// keys and `2B` children.
///
/// Btree vs binary search tree - Binary search tree insertions are "in place". As soon
/// as a leaf node is found which can be parent of the new key, a new node is created.
/// Hence the tree is built top down. In case of btree, the tree is built by splits. Each
/// time a node splits, a new parent is created (or modified). Hence btree is built bottom
/// up.
///
/// ## Add key
/// Adding a key proceeds by finding the leaf node where the key belongs. After adding
/// the key/value, if the invariant of number of keys per node is not satisfied, a node
/// split is performed. The specific logic of node split depends on if a leaf got split
/// or the branch got split or the root iself split.
///
/// * Root - First leaf node is the root. When this leaf gets split a new root is created.
/// New root always has a single key and two children.
///
/// * Leaf - Split of leaf node is handled by creating another leaf node which takes half
/// the keys and values of the current leaf node. Across the 2 leaf nodes, the keys and values
///  remain the same as before split.
///
/// * Branch node - In case of branch node split, the number of keys across the 2 nodes is
/// reduced by one and the key is moved up to the parent as the splitting key. If there were
/// `2B` keys after an insert, there will be `2B + 1` children. After the split, existing node has
/// `B - 1` keys and `B` children, the new split will have `B` keys and `B + 1` children. The extra
/// key is moved up to the as the key. This is needed to ensure that branch nodes continue to
/// have 1 more child than the number of keys.
///
/// ## Remove key
/// Removing a key can violate the invariant by reducing the key count to `B-2` keys where as the
/// minimum required is `B-1`. When this happens, the parent node must handle this by increasing
/// the key count. There are 2 cases to handle -
/// 1. If a neighbor has more than `B` keys, a key can be moved from the neighbor to the current
/// child.
/// 2. If the neighbor also has `B-1` keys (it can't be less to maintain invariant), then the 2
/// nodes need to be merged into a single node. This will reduce a key in the parent which used to
/// be the split key. This might cause an underflow in the parent node and this needs to be handled
/// in the parent's parent.
///
/// The neighbor can be a right neighbor or a left neighbor of the impacted node. In the current
/// implementation, we always consider the left neighbor except for the first child for which
/// right neighbor is considered.
///
/// * Root - If root needs to merge its children when there are only 2 children which are leaf,
/// root is repleaced with the merged leaf reducing the tree level by 1. This is needed to ensure
/// that each branch node has 1 more child than the number of keys.
///
/// * Branch Node - If any child reports undeflow on removing a key, consider following steps -
/// 1. Borrow key from neighbor (left neighbor except for the first child)
/// 2. If borrow key is not possible, merge with neighbor.
/// 3. Remove split key from self. Report underflow if removing violates invariant
///
/// * Leaf Node - Remove key, report underflow if applicable.
///
use log::info;
use serde::Serialize;
use std::cmp;

// Trait bounds for the key of the btree
pub trait NodeKeyType: cmp::Ord + Clone + std::fmt::Display + std::fmt::Debug {}
impl<T: cmp::Ord + Clone + std::fmt::Display + std::fmt::Debug> NodeKeyType for T {}
// Error value returned by btree operations

#[derive(Debug, PartialEq)]
enum BtreeError<K, V> {
    // Error value when an insertion causes a node to split
    NodeSplit(K, Node<K, V>),
    NodeUnderflow(V),
    KeyNotFound,
}

// Represents internal state of a node associated with key K
// Leaf has a value for K and a branch node has associated
// child (could be left or right)
enum KV<K, V> {
    Leaf(K, V),
    Branch(K, Node<K, V>),
}

#[derive(Debug, Serialize, PartialEq)]
enum Node<K, V> {
    Branch(BranchNode<K, V>),
    Leaf(LeafNode<K, V>),
}

impl<K: NodeKeyType, V> Node<K, V> {
    fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
        match self {
            Self::Branch(b) => b.insert(key, value),
            Self::Leaf(l) => l.insert(key, value),
        }
    }
    fn max_key(&self) -> &K {
        match self {
            Self::Branch(b) => b.max_key(),
            Self::Leaf(l) => l.max_key(),
        }
    }
    fn remove_key(&mut self, key: &K) -> Result<V, BtreeError<K, V>> {
        match self {
            Self::Branch(branch) => branch.remove_key(key),
            Self::Leaf(leaf) => leaf.remove_key(key),
        }
    }

    fn find(&self, key: &K) -> Result<&V, BtreeError<K, V>> {
        match self {
            Self::Branch(branch) => branch.find(key),
            Self::Leaf(leaf) => leaf.find(key),
        }
    }

    fn num_keys(&self) -> usize {
        match self {
            Self::Branch(branch) => branch.num_keys(),
            Self::Leaf(leaf) => leaf.num_keys(),
        }
    }

    fn remove_max(&mut self) -> KV<K, V> {
        match self {
            Self::Branch(branch) => {
                let (k, c) = branch.remove_max();
                KV::Branch(k, c)
            }
            Self::Leaf(leaf) => {
                let (k, v) = leaf.remove_max();
                KV::Leaf(k, v)
            }
        }
    }

    fn remove_min(&mut self) -> KV<K, V> {
        match self {
            Self::Branch(branch) => {
                let (k, c) = branch.remove_min();
                KV::Branch(k, c)
            }
            Self::Leaf(leaf) => {
                let (k, v) = leaf.remove_min();
                KV::Leaf(k, v)
            }
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
struct LeafNode<K, V> {
    keys: Vec<K>,
    #[serde(skip)]
    values: Vec<V>,
    #[serde(skip)]
    branch_factor: usize,
}

impl<K: NodeKeyType, V> LeafNode<K, V> {
    fn remove_min(&mut self) -> (K, V) {
        return (self.keys.remove(0), self.values.remove(0));
    }
    fn remove_max(&mut self) -> (K, V) {
        return (self.keys.pop().unwrap(), self.values.pop().unwrap());
    }
    fn new(branch_factor: usize) -> Self {
        LeafNode {
            keys: vec![],
            values: vec![],
            branch_factor: branch_factor,
        }
    }
    fn max_key(&self) -> &K {
        &self.keys[self.keys.len() - 1]
    }

    fn maybe_split(&mut self) -> Result<(), BtreeError<K, V>> {
        if self.keys.len() == self.branch_factor {
            info!("splitting at keys len = {}", self.keys.len());
            let mut split_node = LeafNode::new(self.branch_factor);
            let mid = self.keys.len() / 2;
            split_node.keys = self.keys.split_off(mid);
            split_node.values = self.values.split_off(mid);

            info!(
                "after split current={}, split={}",
                self.keys.len(),
                split_node.keys.len()
            );
            Err(BtreeError::NodeSplit(
                self.keys[mid - 1].clone(),
                Node::Leaf(split_node),
            ))
        } else {
            Ok(())
        }
    }

    fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
        match self.keys.binary_search(&key) {
            Ok(index) => self.values[index] = value,
            Err(index) => {
                self.keys.insert(index, key);
                self.values.insert(index, value);
            }
        }
        self.maybe_split()
    }

    fn remove_key(&mut self, key: &K) -> Result<V, BtreeError<K, V>> {
        match self.keys.binary_search(key) {
            Ok(index) => {
                self.keys.remove(index);
                let value = self.values.remove(index);
                if self.keys.len() < ((self.branch_factor + 1) / 2 - 1) {
                    Err(BtreeError::NodeUnderflow(value))
                } else {
                    Ok(value)
                }
            }
            Err(_) => Err(BtreeError::KeyNotFound),
        }
    }

    fn num_keys(&self) -> usize {
        self.keys.len()
    }
    fn find(&self, key: &K) -> Result<&V, BtreeError<K, V>> {
        match self.keys.binary_search(key) {
            Ok(index) => Ok(&self.values[index]),
            Err(_) => Err(BtreeError::KeyNotFound),
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
struct BranchNode<K, V> {
    keys: Vec<K>,
    children: Vec<Node<K, V>>,
    #[serde(skip)]
    branch_factor: usize,
}

impl<K, V> Into<Node<K, V>> for LeafNode<K, V> {
    fn into(self) -> Node<K, V> {
        Node::Leaf(self)
    }
}

impl<K, V> Into<Node<K, V>> for BranchNode<K, V> {
    fn into(self) -> Node<K, V> {
        Node::Branch(self)
    }
}
impl<K: NodeKeyType, V> BranchNode<K, V> {
    fn remove_min(&mut self) -> (K, Node<K, V>) {
        (self.keys.remove(0), self.children.remove(0))
    }
    fn remove_max(&mut self) -> (K, Node<K, V>) {
        (self.keys.pop().unwrap(), self.children.pop().unwrap())
    }
    fn num_keys(&self) -> usize {
        self.keys.len()
    }
    fn new(branch_factor: usize) -> Self {
        BranchNode {
            keys: vec![],
            children: vec![],
            branch_factor: branch_factor,
        }
    }

    fn max_key(&self) -> &K {
        &self.keys[self.keys.len() - 1]
    }

    fn maybe_split(&mut self) -> Result<(), BtreeError<K, V>> {
        if self.keys.len() == self.branch_factor {
            let mut split_node = BranchNode::new(self.branch_factor);
            let mid = self.keys.len() / 2;
            split_node.keys = self.keys.split_off(mid + 1);
            split_node.children = self.children.split_off(mid + 1);
            Err(BtreeError::NodeSplit(
                self.keys.remove(mid),
                Node::Branch(split_node),
            ))
        } else {
            Ok(())
        }
    }

    fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
        // Each branch node key len is 1 less than children len
        let index_result = self.keys.binary_search(&key);
        let (index, insert_result) = match index_result {
            Ok(index) | Err(index) => (index, self.children[index].insert(key, value)),
        };
        if let Err(BtreeError::NodeSplit(key, split_node)) = insert_result {
            // key is the max key from existing child after the split
            // Case 1: the split child is not the last child.
            //   We replace the key at index with key and insert the split node in child list.
            //   If this reaches the max branching factor, we split the current node.
            // Case 2: the split child is the last child.
            //   We push a new key from the split child. If this reaches the branching factor,
            //   we split the current node.

            if index != self.children.len() - 1 {
                // case 1
                self.keys[index] = key;
                let split_node_key = split_node.max_key();
                match self.keys.binary_search(split_node_key) {
                    Ok(_) => {
                        panic!(
                            "Unexpected key={} in branch={:?}",
                            split_node_key, self.keys
                        );
                    }
                    Err(split_index) => {
                        self.keys.insert(split_index, split_node_key.clone());
                        self.children.insert(split_index, split_node);
                    }
                }
            } else {
                // case 2, set the key as max key of this node and split node as the last child
                self.keys.push(key);
                self.children.push(split_node);
            }
        }
        self.maybe_split()
    }

    fn handle_leaf_undeflow(&mut self, index: usize) {
        // If there's an underflow in leaf, we first check if a key,value pair can be
        // moved to the undeflow node. If not (because it will cause undeflow in the
        // neighbor), we merge the 2 nodes. In either case, the key of the current
        // node changes to account for the move/merge.

        if index > 0 {
            // borrow/merge using left child
            if self.children[index - 1].num_keys() > (self.branch_factor - 1) / 2 {
                match self.children[index - 1] {
                    Node::Leaf(ref mut leaf) => {
                        let (k, v) = leaf.remove_max();
                        let leaf_key = leaf.max_key().clone();
                        self.children[index].insert(k.clone(), v).ok();
                        if index < self.keys.len() - 1 {
                            self.keys[index - 1] = leaf_key;
                        }
                    }
                    _ => panic!("unexpected branch node for leaf child"),
                }
            } else {
                // borrowing from left would cause underflow, need to merge
                let mut underflow_child = self.children.remove(index);
                self.keys.remove(index - 1);

                let left_leaf = match self.children[index - 1] {
                    Node::Leaf(ref mut leaf) => leaf,
                    _ => panic!("expected leaf node"),
                };
                let underflow_leaf = match underflow_child {
                    Node::Leaf(ref mut leaf) => leaf,
                    _ => panic!("expected leaf node"),
                };

                left_leaf.keys.append(&mut underflow_leaf.keys);
                left_leaf.values.append(&mut underflow_leaf.values);
                if index < self.keys.len() - 1 {
                    self.keys[index - 1] = left_leaf.max_key().clone();
                }
            }
        } else {
            // borrow/merge using right child
            if self.children[index + 1].num_keys() > (self.branch_factor - 1) / 2 {
                match self.children[index + 1] {
                    Node::Leaf(ref mut leaf) => {
                        let (k, v) = leaf.remove_min();
                        self.children[index].insert(k.clone(), v).ok();
                        self.keys[index] = k;
                    }
                    _ => panic!("expected leaf node"),
                }
            } else {
                let mut borrow_child = self.children.remove(index + 1);
                self.keys.remove(index);
                let left_leaf = match self.children[index] {
                    Node::Leaf(ref mut leaf) => leaf,
                    _ => panic!("expected leaf node"),
                };
                let borrow_leaf = match borrow_child {
                    Node::Leaf(ref mut leaf) => leaf,
                    _ => panic!("expected leaf node"),
                };
                left_leaf.keys.append(&mut borrow_leaf.keys);
                left_leaf.values.append(&mut borrow_leaf.values);
            }
        }
    }

    fn handle_branch_underflow(&mut self, index: usize) {}

    fn handle_underflow(&mut self, index: usize) {
        match &self.children[index] {
            Node::Leaf(_) => self.handle_leaf_undeflow(index),
            _ => self.handle_branch_underflow(index),
        }
    }

    fn remove_key(&mut self, key: &K) -> Result<V, BtreeError<K, V>> {
        let (index, remove_key_result) = match self.keys.binary_search(key) {
            Ok(index) | Err(index) => (index, self.children[index].remove_key(key)),
        };

        match remove_key_result {
            Err(BtreeError::NodeUnderflow(v)) => {
                self.handle_underflow(index);
                if self.keys.len() < (self.branch_factor - 1) / 2 {
                    Err(BtreeError::NodeUnderflow(v))
                } else {
                    Ok(v)
                }
            }
            res @ _ => res,
        }
    }

    fn find(&self, key: &K) -> Result<&V, BtreeError<K, V>> {
        match self.keys.binary_search(key) {
            Ok(index) | Err(index) => self.children[index].find(key),
        }
    }
}

pub struct BTree<K: cmp::Ord, V> {
    root: Node<K, V>,
    branch_factor: usize, // 2*B - 1
}

impl<K: NodeKeyType, V> BTree<K, V> {
    pub fn new(b: usize) -> Self {
        if b < 2 {
            panic!("{}", "B must be at least 2");
        }
        let branch_factor = 2 * b - 1;
        BTree {
            root: Node::Leaf(LeafNode::new(branch_factor)),
            branch_factor,
        }
    }
    pub fn insert(&mut self, key: K, value: V) {
        match self.root.insert(key, value) {
            Ok(_) => (),
            Err(BtreeError::NodeSplit(key, split_node)) => {
                let branch_node = BranchNode::<K, V>::new(self.branch_factor);
                let old_root = std::mem::replace(&mut self.root, Node::Branch(branch_node));
                let new_root = &mut self.root;
                if let Node::Branch(ref mut node) = new_root {
                    // Every branch node maintains an invariant that it has m keys and
                    // m+1 children.
                    node.keys.push(key);
                    node.children.push(old_root);
                    node.children.push(split_node);
                }
            }
            _ => (),
        }
    }
    pub fn find(&self, key: &K) -> Option<&V> {
        self.root.find(key).ok()
    }
}

#[test]
fn leaf_find_key() {
    let mut leaf = LeafNode::<i32, i32>::new(3);
    leaf.insert(1, 1).ok().unwrap();
    assert!(leaf.find(&1).is_ok());
    assert!(leaf.find(&2).is_err());
}

#[test]
fn leaf_remove_key() {
    let mut leaf = LeafNode::<i32, i32>::new(3);
    leaf.insert(1, 1).ok();
    leaf.insert(2, 2).ok();

    assert!(leaf.find(&2).is_ok());
    leaf.remove_key(&2).ok();
    assert!(leaf.find(&2).is_err());
}

#[test]
fn leaf_remove_key_underflow() {
    let mut leaf = LeafNode::<i32, i32>::new(3);
    leaf.insert(1, 1).ok();
    let ret = leaf.remove_key(&1);
    assert_eq!(BtreeError::<i32, i32>::NodeUnderflow(1), ret.err().unwrap());
}

#[test]
fn leaf_split() {
    let mut leaf = LeafNode::<i32, i32>::new(3);
    leaf.insert(1, 1).ok();
    leaf.insert(2, 2).ok();
    let ret = leaf.insert(3, 3);
    match ret {
        Ok(_) => panic!("{}", "expected not ok on split"),
        Err(BtreeError::NodeSplit(k, Node::Leaf(leaf))) => {
            assert_eq!(k, 1);
            assert_eq!(leaf.keys.len(), 2);
            assert_eq!(leaf.keys, vec![2, 3]);
            assert_eq!(leaf.values, vec![2, 3]);
        }
        a @ _ => panic!("unexpected result = {:?}", a),
    }
}

#[test]
fn branch_split() {
    let mut btree: BTree<i32, i32> = BTree::new(2);
    btree.insert(1, 1);
    btree.insert(2, 2);
    btree.insert(3, 3); // splits into a branch node

    let mut root_node = std::mem::replace(&mut btree.root, Node::Leaf(LeafNode::new(4)));
    let branch_node = match root_node {
        Node::Branch(ref mut b) => b,
        _ => panic!("unexpected node type"),
    };
    branch_node.insert(4, 4).ok(); // leaf node splits and branch is full
    let ret = branch_node.insert(5, 5); // should split branch node
    match ret {
        Ok(_) => panic!("expected error with branch split, got ok"),
        Err(BtreeError::NodeSplit(k, Node::Branch(node))) => {
            assert_eq!(2, k);
            assert_eq!(node.keys.len(), 1);
            assert_eq!(node.children.len(), 2);
            assert_eq!(node.keys, vec![3]);
        }
        a @ _ => panic!("unecpected return value {:?}", a),
    }
}

#[test]
fn branch_find() {
    let mut btree: BTree<i32, i32> = BTree::new(2);
    btree.insert(1, 1);
    btree.insert(2, 2);
    btree.insert(3, 3); // splits into a branch node

    let mut root_node = std::mem::replace(&mut btree.root, Node::Leaf(LeafNode::new(4)));
    let branch_node = match root_node {
        Node::Branch(ref mut b) => b,
        _ => panic!("unexpected node type"),
    };
    branch_node.insert(4, 4).ok(); // leaf node splits and branch is full

    assert!(branch_node.find(&5).is_err());
    assert_eq!(BtreeError::KeyNotFound, branch_node.find(&5).err().unwrap());
    assert_eq!(&2, branch_node.find(&2).ok().unwrap());
}

#[test]
fn branch_handle_underflow_left_child() {
    let mut btree: BTree<i32, i32> = BTree::new(2);
    btree.insert(1, 1);
    btree.insert(2, 2);
    btree.insert(3, 3); // splits into a branch node

    let mut root_node = std::mem::replace(&mut btree.root, Node::Leaf(LeafNode::new(4)));
    let branch_node = match root_node {
        Node::Branch(ref mut b) => b,
        _ => panic!("unexpected node type"),
    };

    assert_eq!(1, branch_node.keys.len());
    let res = branch_node.remove_key(&1);
    assert!(res.is_ok());
    assert!(branch_node.find(&1).is_err());
    assert_eq!(1, branch_node.keys.len());
    assert_eq!(2, branch_node.children.len());
}

#[test]
fn branch_handle_underflow_right_child() {
    let mut btree: BTree<i32, i32> = BTree::new(2);
    btree.insert(1, 1);
    btree.insert(2, 2);
    btree.insert(3, 3); // splits into a branch node
    btree.insert(4, 4); // leaf node splits into [(1), (2), (3, 4)]

    let mut root_node = std::mem::replace(&mut btree.root, Node::Leaf(LeafNode::new(4)));
    let branch_node = match root_node {
        Node::Branch(ref mut b) => b,
        _ => panic!("unexpected node type"),
    };

    assert_eq!(2, branch_node.keys.len());
    assert_eq!(3, branch_node.children.len());

    let res = branch_node.remove_key(&3);
    assert!(res.is_ok());
    let res = branch_node.remove_key(&4);
    assert!(res.is_ok());
    assert!(branch_node.find(&3).is_err());
    assert!(branch_node.find(&4).is_err());

    assert_eq!(1, branch_node.keys.len());
    assert_eq!(2, branch_node.children.len());
}

fn main() {
    let mut bt = BTree::<i32, i32>::new(128);
    for i in 1..1000_000 {
        bt.insert(i, i);
    }
    let before = Instant::now();
    let mut bt1 = BTree::<i32, i32>::new(128);
    for i in 1..1000_000 {
        bt1.insert(i, i);
    }
    println!("elapsed = {} micros", before.elapsed().as_micros());
}

#[test]
fn btree_branch_split() {
    let mut btree: BTree<String, i32> = BTree::new(3);
    btree.insert(String::from("one"), 1);
    btree.insert(String::from("two"), 2);
    btree.insert(String::from("three"), 3);
    btree.insert(String::from("four"), 3);
    btree.insert(String::from("five"), 3);
    btree.insert(String::from("six"), 3);
    btree.insert(String::from("seven"), 3);
    btree.insert(String::from("eight"), 3);
    btree.insert(String::from("nine"), 3);

    assert_eq!(
        "{\"Branch\":{\"keys\":[\"four\",\"seven\"],\"children\":[{\"Leaf\":{\"keys\":[\"eight\",\"five\",\"four\"]}},{\"Leaf\":{\"keys\":[\"nine\",\"one\",\"seven\"]}},{\"Leaf\":{\"keys\":[\"six\",\"three\",\"two\"]}}]}}",
        format!(
            "{}",
            serde_json::to_string(&btree.root).ok().unwrap()
        )
    )
}

#[test]
fn vec_binary_search() {
    let v = vec![10, 100];
    assert_eq!(Ok(0), v.binary_search(&10));
    assert_eq!(Err(0), v.binary_search(&9));
    assert_eq!(Err(1), v.binary_search(&11));
    assert_eq!(Ok(1), v.binary_search(&100));
    assert_eq!(Err(2), v.binary_search(&200));
}
