#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(arbitrary_self_types)]
#![allow(non_upper_case_globals)]
#![feature(hash_set_entry)]

use calcium::Number;
use core::panic;
use gc::{unsafe_empty_trace, Finalize, Gc, GcCell, Trace};
use std::{
    collections::{HashMap, HashSet},
};

macro_rules! simple_empty_finalize_trace {
    ($($T:ty),*) => {
        $(
            impl Finalize for $T {}
            unsafe impl Trace for $T { unsafe_empty_trace!(); }
        )*
    }
}

#[derive(Debug, Trace, Finalize, Clone)]
enum Expression {
    Number(calcium::Number),
    Bool(bool),
    String(String),
    BinOp(Gc<Expression>, BinOp, Gc<Expression>),
    Ident(String),
    Let(Gc<Expression>, Gc<Expression>, Gc<Expression>),
    If(Gc<Expression>, Gc<Expression>, Gc<Expression>),
    Tuple(GcCell<Vec<(Option<Gc<Expression>>, Gc<Expression>)>>),
    List(GcCell<Vec<Gc<Expression>>>),
    Selector(Gc<Expression>, Gc<Expression>),
    Index(Gc<Expression>, Gc<Expression>),
    Lam(Vec<String>, Gc<Expression>, Gc<Type>, Gc<Expression>), // add ident for parameter so that it always looks it up in the context? is the param type necessary?
    Ctl(
        Gc<Expression>, /* The lambda/closure */
        FrameIndex,          /* the index of the starting frame of the delimited continuation */
    ), // an effect handler
    Del(Vec<Frame>), // a delimited continuation
    Closure(Vec<(String, Gc<Expression>)>, Gc<Expression>, Gc<Type>, Gc<Expression>),
    App(Gc<Expression>, Gc<Expression>),
    Fix(Gc<Expression>),
    Case(Gc<Expression>, Vec<(String, String, Gc<Expression>)>), // Compile exhaustive patterns to this, since all we need is an eliminator.
}

#[derive(Debug, Trace, Finalize)]
enum Type {
    None,
    Arr(Gc<Type>, Gc<Type>),
    // TODO
    // tuples, lists, string, bool, number, abs, variant, id
}

impl Type {
    fn arity(&self) -> usize {
        match self {
            Type::None => 0,
            Type::Arr(_, r) => 1 + r.arity(),
        }
    }

    fn new_none() -> Gc<Type> {
        Type::None.into()
    }

    fn new_arr(l: Gc<Type>, r: Gc<Type>) -> Gc<Type> {
        Type::Arr(l, r).into()
    }
}

impl Expression {
    fn new_number(n: Number) -> Gc<Expression> {
        Expression::Number(n).into()
    }
    fn new_int(i: i64) -> Gc<Expression> {
        Expression::new_number(Number::from_clong(i))
    }
    fn new_bool(b: bool) -> Gc<Expression> {
        Expression::Bool(b).into()
    }
    fn new_string(s: &str) -> Gc<Expression> {
        Expression::String(String::from(s)).into()
    }
    fn new_binop(l: Gc<Expression>, op: BinOp, r: Gc<Expression>) -> Gc<Expression> {
        Expression::BinOp(l, op, r).into()
    }
    fn new_ident(s: &str) -> Gc<Expression> {
        Expression::Ident(String::from(s)).into()
    }
    fn new_let(name: Gc<Expression>, v: Gc<Expression>, x: Gc<Expression>) -> Gc<Expression> {
        Expression::Let(name, v, x).into()
    }
    fn new_slet(name: &str, v: Gc<Expression>, x: Gc<Expression>) -> Gc<Expression> {
        Expression::new_let(Expression::new_ident(name), v, x)
    }

    fn new_if(cond: Gc<Expression>, then: Gc<Expression>, els: Gc<Expression>) -> Gc<Expression> {
        Expression::If(cond, then, els).into()
    }
    fn new_tuple(v: GcCell<Vec<(Option<Gc<Expression>>, Gc<Expression>)>>) -> Gc<Expression> {
        Expression::Tuple(v).into()
    }
    fn new_list(v: GcCell<Vec<Gc<Expression>>>) -> Gc<Expression> {
        Expression::List(v).into()
    }
    fn new_selector(x: Gc<Expression>, field: Gc<Expression>) -> Gc<Expression> {
        Expression::Selector(x, field).into()
    }
    fn new_index(x: Gc<Expression>, i: Gc<Expression>) -> Gc<Expression> {
        Expression::Index(x, i).into()
    }
    fn new_lam(free_vars: Vec<String>, bind: Gc<Expression>, ty: Gc<Type>, body: Gc<Expression>) -> Gc<Expression> {
        Expression::Lam(free_vars, bind, ty, body).into()
    }
    fn new_tlam(bind: &str, ty: Gc<Type>, body: Gc<Expression>) -> Gc<Expression> {
        Expression::new_lam(Vec::new(), Expression::new_ident(bind), ty, body)
    }
    fn new_slam(bind: &str, body: Gc<Expression>) -> Gc<Expression> {
        Expression::new_lam(Vec::new(), Expression::new_ident(bind), Type::None.into(), body)
    }
    fn new_app(fun: Gc<Expression>, arg: Gc<Expression>) -> Gc<Expression> {
        Expression::App(fun, arg).into()
    }
    fn new_closure(
        env: Vec<(String, Gc<Expression>)>,
        bind: Gc<Expression>,
        ty: Gc<Type>,
        body: Gc<Expression>,
    ) -> Gc<Expression> {
        Expression::Closure(env, bind, ty, body).into()
    }
    fn new_fix(x: Gc<Expression>) -> Gc<Expression> {
        Expression::Fix(x).into()
    }

    fn new_ctl(lam: Gc<Expression>, frame_index: FrameIndex) -> Gc<Expression> {
        Expression::Ctl(lam, frame_index).into()
    }

    fn new_del(del: Vec<Frame>) -> Gc<Expression> {
        Expression::Del(del).into()
    }
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    CmpEq,
    // Assign,
    // Comparison,
}

simple_empty_finalize_trace![BinOp];

#[derive(Debug, Trace, Finalize, Clone)]
enum Frame {
    Rv(Option<Gc<Expression>>),
    Rec {
        expr: Gc<Expression>,
        locals: GcCell<HashMap<String, Gc<Expression>>>,
        rv: Option<Gc<Expression>>,
    },
}

impl Frame {
    fn new_rec(x: Gc<Expression>) -> Frame {
        Frame::Rec {
            expr: x,
            locals: GcCell::new(HashMap::new()),
            rv: None,
        }
    }
    fn new_rec_hm(x: Gc<Expression>, locals: HashMap<String, Gc<Expression>>) -> Frame {
        Frame::Rec {
            expr: x,
            locals: GcCell::new(locals),
            rv: None,
        }
    }
    fn new_rec_locals<const N: usize>(
        x: Gc<Expression>,
        locals: [(String, Gc<Expression>); N],
    ) -> Frame {
        Frame::Rec {
            expr: x,
            locals: GcCell::new(HashMap::from(locals)),
            rv: None,
        }
    }
}

fn set_rv(stack: &mut Vec<Frame>, v: Gc<Expression>) {
    stack.pop();
    match stack.last_mut() {
        Some(Frame::Rv(opt)) => opt.insert(v),
        Some(Frame::Rec { expr: _, locals: _, rv }) => rv.insert(v),
        None => panic!("expected frame to insert return value"),
    };
}

fn lookup_id(stack: &Vec<Frame>, s: &str) -> Gc<Expression> {
    for frame in stack.iter().rev() {
        if let Frame::Rec {
            expr: _,
            locals,
            rv: _,
        } = frame
        {
            // println!("looking up {} in {:?}", s, locals.borrow().keys());
            if let Some(x) = locals.borrow().get(s) {
                return x.clone();
            }
        }
    }
    panic!("variable {s} was not in scope")
}

fn add_fresh_name(unique_names: &mut HashSet<String>, x: &str) -> String {
    let mut num = 1;
    let mut to_check = x.to_owned();
    while unique_names.contains(&to_check as &str) {
        to_check = format!("{x}{num}");
        num += 1;
    }
    unique_names.insert(to_check.clone());
    to_check
}

fn merge_free_vars(dst: &mut Vec<String>, subterm: &Gc<Expression>, old_bind: &str) {
    match &**subterm {
        Expression::Lam(subterm_free_vars, _, _, _) => {
            // for each variable in subterm that is not old_bind, add it to dst
            for x in subterm_free_vars.iter() {
                if x != old_bind && !dst.contains(x) {
                    dst.push(x.clone());
                }
            }
        },
        Expression::Let(bind, rhs, body) => {
            let Expression::Ident(bind) = &**bind else { panic!("expected ident") };
            merge_free_vars(dst, rhs, bind);
            merge_free_vars(dst, body, bind);
            dst.retain(|x| x != bind && x != old_bind);
        },
        _ => (),
    }
}

// adjust resolve_names logic to do FV(term) = FV(subterm) - {introduced variables}
fn resolve_names(
    m: &mut Vec<(String, String)>,
    free_vars: &mut Vec<String>,
    unique_names: &mut HashSet<String>,
    dist: usize,
    prog: &Gc<Expression>,
) -> Gc<Expression> {
    match &**prog {
        Expression::BinOp(l, op, r) => {
            let l = resolve_names(m, free_vars, unique_names, dist, l);
            let r = resolve_names(m, free_vars, unique_names, dist, r);
            Expression::new_binop(l, *op, r)
        }
        Expression::Let(bind, rhs, body) => {
            let rhs = resolve_names(m, free_vars, unique_names, dist, rhs);
            let Expression::Ident(x) = &**bind else { panic!("expected identifier in let binding") };
            let fresh_name = add_fresh_name(unique_names, x);
            m.push((x.clone(), fresh_name.clone()));
            let body = resolve_names(m, free_vars, unique_names, dist + 1, body);
            m.pop();
            unique_names.remove(&fresh_name);
            Expression::new_let(Expression::new_ident(&fresh_name), rhs, body)
        }
        Expression::If(cond, then, els) => {
            let cond = resolve_names(m, free_vars, unique_names, dist, cond);
            let then = resolve_names(m, free_vars, unique_names, dist, then);
            let els = resolve_names(m, free_vars, unique_names, dist, els);
            Expression::new_if(cond, then, els)
        }
        Expression::Tuple(t) => {
            let v : Vec<(Option<Gc<Expression>>, Gc<Expression>)> = 
                t.borrow().iter().map(|(a, b)| (a.clone(), resolve_names(m, free_vars, unique_names, dist, b))).collect();
            Expression::new_tuple(GcCell::new(v))
        },
        Expression::List(l) => {
            // we're not mutating in place. we're creating a new vector.
            let v: Vec<Gc<Expression>> = l.borrow().iter().map(|e| resolve_names(m, free_vars, unique_names, dist, e)).collect();
            Expression::new_list(GcCell::new(v))
        },
        Expression::Selector(x, sel) => {
            let x = resolve_names(m, free_vars, unique_names, dist, x);
            Expression::new_selector(x, sel.clone())
        },
        Expression::Index(col, ind) => {
            let col = resolve_names(m, free_vars, unique_names, dist, col);
            let ind = resolve_names(m, free_vars, unique_names, dist, ind);
            Expression::new_index(col, ind)
        },
        Expression::Ident(x) => {
            // look up x in m up to dist
            // if not found, then it's a free variable
            // append to free_vars
            let mut i = 0;
            let mut s: String = x.to_string();
            let mut is_local = false;
            for var in m.iter().rev() {
                if &var.0 == x {
                    is_local = i <= dist;
                    // Look up the unique name for this binding and replace it with that.
                    s = var.1.clone();
                    break;
                }
                i += 1;
            }
            if !is_local {
                free_vars.push(s.to_string());
            }
            return Expression::new_ident(&s);
        }
        Expression::Lam(_, bind, /*TODO: resolve names in types */ ty, body) => {
            let Expression::Ident(x) = &**bind else { panic!("expected identifier in lambda binding") };
            let fresh_name = add_fresh_name(unique_names, x);
            m.push((x.clone(), fresh_name.clone()));
            let mut lam_free_vars = Vec::new();
            let body = resolve_names(m, &mut lam_free_vars, unique_names, 0, body);
            m.pop();
            unique_names.remove(&fresh_name);
            merge_free_vars(&mut lam_free_vars, &body, &fresh_name);
            Expression::new_lam(lam_free_vars, Expression::new_ident(&fresh_name), ty.clone(), body)
        }
        Expression::App(fun, arg) => {
            let fun = resolve_names(m, free_vars, unique_names, dist, fun);
            let arg = resolve_names(m, free_vars, unique_names, dist, arg);
            Expression::new_app(fun, arg)
        }
        Expression::Fix(x) => {
            Expression::new_fix(resolve_names(m, free_vars, unique_names, dist, x))
        }
        Expression::Case(_, _) => todo!(),
        Expression::Ctl(lam, i) => {
            Expression::new_ctl(resolve_names(m, free_vars, unique_names, dist, lam), i.clone())
        }
        _ => prog.clone(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FrameIndex {
    Unset,
    Finished,
    Set(usize),
}

simple_empty_finalize_trace!(FrameIndex);

fn dump_frame_index(x: &Gc<Expression>) {
    if let Expression::Ctl(_, frame_index) = &**x {
        println!("frame_index={:?}", frame_index);
    }
}

fn arity(x: &Gc<Expression>) -> usize {
    match &**x {
        Expression::Closure(_, _, ty, _) => ty.arity(),
        Expression::Lam(_, _, ty, _) => ty.arity(),
        Expression::Fix(_) => todo!(),
        _ => panic!("not a function"),
    }
}

fn eval_with_stack(mut stack: Vec<Frame>) -> Gc<Expression> {
    while let Some(frame) = stack.last() {
        let mut to_capture : Option<Gc<Expression>> = None;
        let mut to_set: Option<Gc<Expression>> = None;
        let mut to_push: Option<Frame> = None;
        let mut to_restore : Option<Gc<Expression>> = None;
        match frame {
            Frame::Rv(Some(rv)) => return rv.clone(),
            Frame::Rv(None) => panic!("expected return value at the top of the stack"),
            Frame::Rec { expr, locals, rv } => match &**expr {
                Expression::Number(_) => to_set = Some(expr.clone()),
                Expression::Bool(_) => to_set = Some(expr.clone()),
                Expression::String(_) => to_set = Some(expr.clone()),
                Expression::Closure(_, _, _, _) => to_set = Some(expr.clone()),
                Expression::Ident(s) => {
                    // println!("lookup id={:?}", s);
                    let val = lookup_id(&stack, s);
                    to_set = Some(val)
                },
                Expression::BinOp(l, op, r) => match rv {
                    None => to_push = Some(Frame::new_rec(l.clone())),
                    Some(v) => {
                        // TODO: throw exception effect if divide by zero
                        let mut locals = locals.borrow_mut();
                        match locals.get("1") {
                            None => {
                                locals.insert("1".to_string(), v.clone());
                                to_push = Some(Frame::new_rec(r.clone()));
                            },
                            Some(v1) => {
                                let res = match (&**v1, op, &**v) {
                                    (Expression::Number(l), BinOp::Add, Expression::Number(r)) => Expression::new_number(l+r),
                                    (Expression::Number(l), BinOp::Sub, Expression::Number(r)) => Expression::new_number(l-r),
                                    (Expression::Number(l), BinOp::Mul, Expression::Number(r)) => Expression::new_number(l*r),
                                    (Expression::Number(l), BinOp::Div, Expression::Number(r)) => Expression::new_number(l/r),
                                    (Expression::Number(l), BinOp::CmpEq, Expression::Number(r)) => Expression::new_bool(l==r),
                                    (Expression::String(l), BinOp::Add, Expression::String(r)) => Expression::new_string(&(l.clone() + r)),
                                    (Expression::String(l), BinOp::CmpEq, Expression::String(r)) => Expression::new_bool(l==r),
                                    _ => panic!("left and right were not numbers"),
                                };
                                to_set = Some(res);
                            },
                        }
                    },
                },
                Expression::Let(id, x, body) if let Expression::Ident(id) = &**id => match rv {
                    None => to_push = Some(Frame::new_rec(x.clone())),
                    Some(v) => {
                        // println!("let id={:?}", id);
                        let mut locals = locals.borrow_mut();
                        if !locals.contains_key("1") {
                            let mut res: Gc<Expression> = v.clone();
                            if let Expression::Ctl(cls, FrameIndex::Unset) = &**v {
                                // if RHS is a Ctl, then we set its delimiter
                                // This is like the reset of a shift/reset.
                                // By constructing a handler, we set a delimiter on the stack.
                                // Our approach right now will be to copy the part of the stack.
                                // The delimiter will be that of the body, so it will be 1+the index of the current frame.
                                res = Expression::new_ctl(cls.clone(), FrameIndex::Set(stack.len()));
                            }
                            locals.insert("1".to_string(), res.clone());
                            to_push = Some(Frame::new_rec_locals(body.clone(), [(id.clone(), res)]));
                        } else {
                            let mut res = v.clone();
                            if let Expression::Ctl(cls, FrameIndex::Set(_)) = &**v {
                                res = Expression::new_ctl(cls.clone(), FrameIndex::Finished);
                            }
                            to_set = Some(res);
                            // dump_frame_index(&v);
                        }
                    },
                },
                Expression::Let(_, _, _) => todo!(),
                Expression::If(cond, then, els) => match rv {
                    None => to_push = Some(Frame::new_rec(cond.clone())),
                    Some(v) => {
                        let mut locals = locals.borrow_mut();
                        if locals.contains_key("1") {
                            to_set = Some(v.clone());
                        } else {
                            locals.insert("1".to_string(), v.clone());
                            let Expression::Bool(b) = **v else { panic!("expected boolean") };
                            to_push = if b {
                                Some(Frame::new_rec(then.clone()))
                            } else {
                                Some(Frame::new_rec(els.clone()))
                            }
                        }
                    },
                },
                Expression::Tuple(t) => match rv {
                    None => {
                        let mut locals = locals.borrow_mut();
                        let t = t.borrow();
                        locals.insert("1".to_string(), Expression::new_tuple(GcCell::new(Vec::with_capacity(t.len()))));
                        if let Some((_, b)) = t.first() {
                            to_push = Some(Frame::new_rec(b.clone()));
                        }
                    },
                    Some(rv) => {
                        let locals = locals.borrow();
                        let vc = locals.get("1").unwrap();
                        let Expression::Tuple(dst) = &**vc else { panic!("expected tuple") };
                        let mut dst = dst.borrow_mut();
                        let prev_i = dst.len();
                        let t = t.borrow();
                        let a = &t[prev_i].0;
                        dst.push((a.clone(), rv.clone()));
                        let i = dst.len();
                        if i == t.len() {
                            to_set = Some(vc.clone());
                        } else {
                            let (_, b) = &t[i];
                            to_push = Some(Frame::new_rec(b.clone()));
                        }
                    },
                },
                Expression::List(l) => match rv {
                    None => {
                        let mut locals = locals.borrow_mut();
                        let l = l.borrow();
                        locals.insert("1".to_string(), Expression::new_list(GcCell::new(Vec::with_capacity(l.len()))));
                        if let Some(elem) = l.first() {
                            to_push = Some(Frame::new_rec(elem.clone()));
                        }
                    },
                    Some(rv) => {
                        let locals = locals.borrow();
                        let vc = locals.get("1").unwrap();
                        let Expression::List(dst) = &**vc else { panic!("expected list") };
                        let mut dst = dst.borrow_mut();
                        dst.push(rv.clone());
                        let i = dst.len();
                        if i == l.borrow().len() {
                            to_set = Some(vc.clone());
                        } else {
                            let elem = &l.borrow()[i];
                            to_push = Some(Frame::new_rec(elem.clone()));
                        }
                    },
                },
                Expression::Selector(t, sel) => match rv {
                    None => to_push = Some(Frame::new_rec(t.clone())), // eval tuple
                    Some(rv) => {
                        let Expression::Tuple(t) = &**rv else { panic!("expected tuple") };
                        let Expression::Ident(sel) = &**sel else { panic!("expected ident") };
                        let found = t.borrow().iter().find_map(|(a, b)| {
                            if let Some(got_sel) = a && let Expression::Ident(got_sel) = &**got_sel && got_sel == sel {
                                Some(b)
                            } else { None }
                        }).unwrap_or_else(|| panic!("selector {} not found", sel)).clone();
                        to_set = Some(found);
                    },
                },
                Expression::Index(col, ind) => match rv {
                    None => to_push = Some(Frame::new_rec(col.clone())), // eval collection
                    Some(rv) => if !locals.borrow().contains_key("1") {
                        // eval ind
                        locals.borrow_mut().insert("1".to_string(), rv.clone());
                        to_push = Some(Frame::new_rec(ind.clone()));
                    } else {
                        // index into collection
                        let locals = locals.borrow();
                        let Some(col) = locals.get("1") else { panic!("expected collection") };
                        let ind : usize = {
                            let Expression::Number(ind) = &**rv else { panic!("expected number") };
                            ind.into()
                        };
                        let res = match &**col {
                            Expression::List(l) => l.borrow()[ind].clone(),
                            Expression::Tuple(t) => t.borrow()[ind].1.clone(),
                            _ => panic!("expected collection"),
                        };
                        to_set = Some(res);
                    },
                },
                Expression::Lam(free_vars, bind, ty, body) => {
                    // For each free variable, look it up in the environment and create closure object with it.
                    let mut env = Vec::with_capacity(free_vars.len());
                    for name in free_vars {
                        let val = lookup_id(&stack, name);
                        env.push((name.clone(), val));
                    }
                    to_set = Some(Expression::new_closure(env, bind.clone(), ty.clone(), body.clone()))
                },
                Expression::App(fun, arg) => match rv {
                    // eval fun
                    None => to_push = Some(Frame::new_rec(fun.clone())),                    
                    Some(rv) =>
                    if !locals.borrow().contains_key("2") {
                        if !locals.borrow().contains_key("1") {
                            // eval arg
                            locals.borrow_mut().insert("1".to_string(), rv.clone());
                            to_push = Some(Frame::new_rec(arg.clone()));
                        } else {
                            // match against evaluated fun. eval body
                            let v1 : Gc<Expression> = {
                                let mut locals = locals.borrow_mut();
                                locals.insert("2".to_string(), rv.clone());
                                locals.remove("1").unwrap()
                            };
                            match &*v1 {
                                Expression::Del(_) => to_restore = Some(v1.clone()),
                                Expression::Ctl(cls, fi) => {
                                    /* if arity of cls's type is 2, then we capture instead of setting a return value */
                                    locals.borrow_mut().insert("3".to_string(), v1.clone());
                                    to_push = Some(Frame::new_rec(Expression::new_app(cls.clone(), rv.clone())));
                                    // to_capture = Some(v1)
                                },
                                _ => {
                                    let Expression::Closure(env, bind, _, body) = &*v1 else { panic!("expected closure, got {:?}", v1) };
                                    let Expression::Ident(bind) = &**bind else { panic!("expected ident") };
                                    let mut fn_locals: HashMap<String, Gc<Expression>> = HashMap::new();
                                    for (id, val) in env {
                                        fn_locals.insert(id.to_string(), val.clone());
                                    }
                                    fn_locals.insert(bind.to_string(), rv.clone());
                                    to_push = Some(Frame::new_rec_hm(body.clone(), fn_locals));
                                }
                            }                                 
                        }
                    } else {
                        let mut locals = locals.borrow_mut();
                        if let Some(x) = locals.remove("3") && let Expression::Ctl(old_cls, fi) = &*x {
                            let res = Expression::new_ctl(rv.clone(), fi.clone());
                            // depending on the arity of rv, we either capture or set a return value
                            if arity(&rv) == 1 {
                                to_capture = Some(res);
                            } else {
                                to_set = Some(res);
                            }
                        } else {
                        // if let Expression::Ctl(old_cls, fi) = 
                        // return evaluated body
                            to_set = Some(rv.clone());
                        }
                    },
                },
                /*
                Fix(t) -> if t.is_val()
                    then Lam("$v", App(App(t, Fix(t)), "$v"))
                    else Fix(eval(t))
                */
                Expression::Fix(x) => {
                    match rv {
                    None => to_push = Some(Frame::new_rec(x.clone())),
                    Some(rv) => {
                        if !locals.borrow().contains_key("1") {
                            locals.borrow_mut().insert("1".to_string(), rv.clone());
                            // arity of fixed function?
                            let lam_type = match &**rv {
                                Expression::Closure(_, _, ty, _) if let Type::Arr(from, _) = &**ty => from.clone(),
                                Expression::Lam(_, _, ty, _) if let Type::Arr(from, _) = &**ty => from.clone(),
                                _ => panic!("not a function"),
                            };
                            let z_comb = Expression::new_tlam(
                                "$v",
                                lam_type,
                                Expression::new_app(
                                    Expression::new_app(rv.clone(), Expression::new_fix(rv.clone())),
                                    Expression::new_ident("$v")),
                            );
                            to_push = Some(Frame::new_rec(z_comb));
                        } else {
                            to_set = Some(rv.clone());
                        }
                    },
                }},
                Expression::Case(_, _) => todo!(),
                // Expression::Ctl(_, _, _, _, _) => todo!(),
                Expression::Ctl(fun, frame_index) => match rv {
                    // Evaluate fun to a closure
                    // Return a Ctl expression with the closure and the frame index
                    None => to_push = Some(Frame::new_rec(fun.clone())),
                    Some(cls) => {
                        let Expression::Closure(_, _, _, _) = &**cls else { panic!("expected closure") };
                        to_set = Some(Expression::new_ctl(cls.clone(), frame_index.clone()))
                    },
                },
                Expression::Del(_) => to_set = Some(expr.clone()),
            },
        }
        if let Some(Expression::Del(v)) = to_restore.as_deref() {
            let arg = {
                let Some(Frame::Rec { expr: _, locals, rv: _ }) = stack.last() else { panic!("expected record")};
                locals.borrow().get("2").unwrap().clone()
            };
            stack.extend(v.iter().map(Frame::to_owned));
            if let Some(Frame::Rec { expr: _, locals: _, rv }) = stack.last_mut() {
                _ = rv.insert(arg);
            }
        }
        if let Some(Expression::Ctl(cls, del)) = to_capture.as_deref() {
            let FrameIndex::Set(del) = *del else { panic!("handler out of scope") }; // TODO: make this an effect as well
            let k : Vec<Frame> = stack.drain(del..).collect();
            let next_exp = Expression::new_app(cls.clone(), Expression::new_del(k));
            to_push = Some(Frame::new_rec(next_exp));
        }
        if let Some(x) = to_set {
            set_rv(&mut stack, x);
        }
        if let Some(f) = to_push {
            stack.push(f);
        }
    }
    unreachable!()
}

fn eval(prog: Gc<Expression>) -> Gc<Expression> {
    let mut m = Vec::new();
    let mut free_vars = Vec::new();
    let mut unique_names = HashSet::new();
    let prog = resolve_names(&mut m, &mut free_vars, &mut unique_names, 0, &prog);
    eval_with_stack(vec![Frame::Rv(None), Frame::new_rec(prog)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number() {
        println!("evaluating 1");
        let prog = Expression::new_int(1);
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(1)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_bool() {
        println!("evaluating true");
        let prog = Expression::new_bool(true);
        let res = eval(prog);
        match &*res {
            Expression::Bool(b) => assert_eq!(b, &true),
            _ => self::panic!("expected bool"),
        }
    }

    #[test]
    fn test_string() {
        println!(r#"evaluating "hello""#);
        let prog = Expression::new_string("hello");
        let res = eval(prog);
        match &*res {
            Expression::String(s) => assert_eq!(s, "hello"),
            _ => self::panic!("expected string"),
        }
    }

    #[test]
    fn test_sub() {
        println!("evaluating 1-1");
        let prog = Expression::new_binop(Expression::new_int(1), BinOp::Sub, Expression::new_int(1));
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(0)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_mul() {
        println!("evaluating 2*3");
        let prog = Expression::new_binop(Expression::new_int(2), BinOp::Mul, Expression::new_int(3));
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(6)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_div() {
        println!("evaluating 6/2");
        let prog = Expression::new_binop(Expression::new_int(6), BinOp::Div, Expression::new_int(2));
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(3)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_cmp_eq() {
        println!("evaluating 1==1");
        let prog = Expression::new_binop(Expression::new_int(1), BinOp::CmpEq, Expression::new_int(1));
        let res = eval(prog);
        match &*res {
            Expression::Bool(b) => assert_eq!(b, &true),
            _ => self::panic!("expected bool"),
        }
    }

    #[test]
    fn test_list_index() {
        println!("evaluating list(1+1, 2+2, 3+3)[1]");
        let prog = Expression::new_index(
            Expression::new_list(GcCell::new(vec![
                Expression::new_binop(Expression::new_int(1), BinOp::Add, Expression::new_int(1)),
                Expression::new_binop(Expression::new_int(2), BinOp::Add, Expression::new_int(2)),
                Expression::new_binop(Expression::new_int(3), BinOp::Add, Expression::new_int(3)),
            ])), 
            Expression::new_int(1),
        );
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(4)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_tuple_index() {
        println!("evaluating (1+1, 2+2, 3+3)[1]");
        let prog = Expression::new_index(
            Expression::new_tuple(GcCell::new(vec![
                (None, Expression::new_binop(Expression::new_int(1), BinOp::Add, Expression::new_int(1))),
                (None, Expression::new_binop(Expression::new_int(2), BinOp::Add, Expression::new_int(2))),
                (None, Expression::new_binop(Expression::new_int(3), BinOp::Add, Expression::new_int(3))),
            ])), 
            Expression::new_int(1),
        );
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(4)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_tuple_selector() {
        println!("evaluating (1+1, b: 2+2, 3+3).b");
        let prog = Expression::new_selector(
            Expression::new_tuple(GcCell::new(vec![
                (None, Expression::new_binop(Expression::new_int(1), BinOp::Add, Expression::new_int(1))),
                (Some(Expression::new_ident("b")), Expression::new_binop(Expression::new_int(2), BinOp::Add, Expression::new_int(2))),
                (None, Expression::new_binop(Expression::new_int(3), BinOp::Add, Expression::new_int(3))),
            ])), 
            Expression::new_ident("b"),
        );
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(4)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_effect_resume() {
        println!(r#"evaluating
        let x = ctl(fun arg1 arg2 k -> k(arg1 + arg2))
        in x 1 2 + 3
        "#);
        let prog = Expression::new_let(
            Expression::new_ident("x"),
            Expression::new_ctl(Expression::new_tlam(
                "arg1", 
                Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_none()))), 
                Expression::new_tlam(
                    "arg2", 
                    Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_none())),
                    Expression::new_tlam(
                        "k",
                        Type::new_arr(Type::new_none(), Type::new_none()),
                        Expression::new_app(Expression::new_ident("k"), Expression::new_binop(Expression::new_ident("arg1"), BinOp::Add, Expression::new_ident("arg2"))))
                    )),
            FrameIndex::Unset),
            Expression::new_binop(Expression::new_app(Expression::new_app(Expression::new_ident("x"), Expression::new_int(1)), Expression::new_int(2)), BinOp::Add, Expression::new_int(3))
        );
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(6)),
            _ => self::panic!("expected number"),
        }
    }

    #[test]
    fn test_effect_abortive() {
        println!(r#"evaluating
        let x = ctl(fun arg k -> arg)
        in x(1) + 2
        "#);
        let prog = Expression::new_let(
            Expression::new_ident("x"),
            Expression::new_ctl(Expression::new_tlam("arg", Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_none())), Expression::new_tlam("k", Type::new_arr(Type::new_none(), Type::new_none()), Expression::new_ident("arg"))),FrameIndex::Unset),
            Expression::new_binop(Expression::new_app(Expression::new_ident("x"), Expression::new_int(1)), BinOp::Add, Expression::new_int(2))
        );
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(1)),
            _ => self::panic!("expected number"),
        }  
    }

    #[test]
    fn test_fix() {
        println!(r#"evaluating
        let sum = fix (fun f n -> if n = 0 then 0 else n + f(n-1))
        in sum(3)
        "#);
        let prog = Expression::new_slet(
            "sum",
            Expression::new_fix(
                Expression::new_tlam(
                    "f",
                    Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_none())),
                    Expression::new_tlam(
                        "n",
                        Type::new_arr(Type::new_none(), Type::new_none()),
                        Expression::new_if(
                            Expression::new_binop(Expression::new_ident("n"), BinOp::CmpEq, Expression::new_int(0)),
                            Expression::new_int(0),
                            Expression::new_binop(
                                Expression::new_ident("n"),
                                BinOp::Add,
                                Expression::new_app(
                                    Expression::new_ident("f"),
                                    Expression::new_binop(Expression::new_ident("n"), BinOp::Sub, Expression::new_int(1))
                                )
                            )
                        )
                    )
                )
            ),
            Expression::new_app(Expression::new_ident("sum"), Expression::new_int(3)),
        );
        let res = eval(prog);
        match &*res {
            Expression::Number(n) => assert_eq!(n, &Number::from_ulong(6)),
            _ => self::panic!("expected number"),
        }
    }
}

fn main() {
    // let prog = Expression::new_let(
    //     Expression::new_ident("x"),
    //     Expression::new_number(Number::from_ulong(3)),
    //     Expression::new_binop(
    //         Expression::new_number(Number::one()),
    //         BinOp::Add,
    //         Expression::new_ident("x"),
    //     ),
    // );
    // let x = Expression::new_binop(Expression::new_number(Number::one()), BinOp::Add, Expression::new_number(Number::from_ulong(2)));
    // let prog = Expression::new_if(Expression::new_let(Expression::new_ident("x"), Expression::new_bool(false), Expression::new_ident("x")), Expression::new_number(Number::from_ulong(3)), Expression::new_number(Number::from_ulong(4)));
    // let prog = Expression::new_app(
    //     Expression::new_abs(
    //         Expression::new_ident("x"),
    //         Type::None.into(), // don't care about type
    //         Expression::new_ident("x"),
    //     ),
    //     Expression::new_number(Number::from_ulong(3)),
    // );
    // let prog = Expression::new_app(
    //     Expression::new_let(
    //         Expression::new_ident("l"),
    //         Expression::new,
    //         _),
    //     Expression::new_number(Number::from_ulong(2)),
    // );
    // let prog = Expression::new_let(
    //     Expression::new_ident("x"),
    //     Expression::new_let(
    //         Expression::new_ident("x"),
    //         Expression::new_number(Number::one()),
    //         Expression::new_abs(
    //             Expression::new_ident("y"),
    //             Type::None.into(),
    //             Expression::new_binop(Expression::new_ident("y"), BinOp::Add, Expression::new_ident("x")))),
    //     Expression::new_app(Expression::new_ident("x"), Expression::new_number(Number::from_ulong(2))),
    // );
    // let prog = Expression::new_let(
    //     Expression::new_ident("x"),
    //     Expression::new_number(Number::from_ulong(1)),
    //     Expression::new_let(
    //         Expression::new_ident("z"),
    //         Expression::new_number(Number::from_ulong(2)),
    //         Expression::new_let(
    //             Expression::new_ident("y"),
    //             Expression::new_lam(
    //                 Expression::new_ident("x"),
    //                 Type::None.into(),
    //                 Expression::new_binop(
    //                     Expression::new_ident("x"),
    //                     BinOp::Add,
    //                     Expression::new_ident("z"),
    //                 ),
    //             ),
    //             Expression::new_app(
    //                 Expression::new_ident("y"),
    //                 Expression::new_number(Number::from_ulong(2)),
    //             ),
    //         ),
    //     ),
    // );


    // println!(r#"evaluating
    // let x = ctl(fun arg k -> k(arg))
    // in x(1) + 2
    // "#);
    // let prog = Expression::new_let(
    //     Expression::new_ident("x"),
    //     Expression::new_ctl(Expression::new_slam("arg", Expression::new_slam("k", Expression::new_app(Expression::new_ident("k"), Expression::new_ident("arg")))),FrameIndex::Unset),
    //     Expression::new_binop(Expression::new_app(Expression::new_ident("x"), Expression::new_int(1)), BinOp::Add, Expression::new_int(2))
    //     // Expression::new_ctl(Expression::new_lam(Expression::new_ident("arg"), Type::None.into(), Expression::new_lam(Expression::new_ident("k"), Type::None.into(), Expression::new_app(Expression::new_ident("k"), Expression::new_ident("arg")))), GcCell::new(0)),
    //     // Expression::new_binop(Expression::new_app(Expression::new_ident("x"), Expression::new_number(Number::one())), BinOp::Add, Expression::new_number(Number::from_ulong(2)))
    // );
    
    // println!(r#"evaluating
    // list(1+1, 2+2, 3+3)
    // "#);
    // let prog = 
    //     Expression::new_list(GcCell::new(vec![
    //         Expression::new_binop(Expression::new_int(1), BinOp::Add, Expression::new_int(1)),
    //         Expression::new_binop(Expression::new_int(2), BinOp::Add, Expression::new_int(2)),
    //         Expression::new_binop(Expression::new_int(3), BinOp::Add, Expression::new_int(3)),
    //     ]));

    // println!(r#"evaluating
    // let x =
    //     let y = ctl(fun arg k -> arg)
    //     in y
    // in x(1)
    // "#);
    // let prog = Expression::new_slet(
    //     "x",
    //     Expression::new_slet(
    //         "y",
    //         Expression::new_ctl(Expression::new_slam("arg", Expression::new_slam("k", Expression::new_ident("arg"))), FrameIndex::Unset),
    //         Expression::new_ident("y"),
    //     ),
    //     Expression::new_app(Expression::new_ident("x"), Expression::new_int(1))
    // );

    // println!("{:?}", &prog);
    // let mut m = Vec::new();
    // let mut free_vars = Vec::new();
    // let mut unique_names = HashSet::new();
    // let prog = resolve_names(&mut m, &mut free_vars, &mut unique_names, 0, &prog);
    // println!("{:?}", &prog);
    // let stackres = eval_stack(prog);
    // println!("stack res = {:?}", stackres);

    // let prog = Expression::new_selector(
    //     Expression::new_tuple(vec![(
    //         Some(Expression::new_ident("x")),
    //         Expression::new_number(2),
    //     )]),
    //     Expression::new_ident("x"),
    // );
    // let prog = Expression::new_index(
    //     Expression::new_tuple(vec![(
    //         Some(Expression::new_ident("x")),
    //         Expression::new_number(2),
    //     )]),
    //     Expression::new_number(0),
    // );
    let prog = Expression::new_tlam(
        "w",
        Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_none()))),
        Expression::new_tlam(
            "x",
            Type::new_arr(Type::new_none(), Type::new_arr(Type::new_none(), Type::new_none())),
            Expression::new_slet(
                "y", 
                Expression::new_int(3),
                Expression::new_tlam(
                    "z", 
                    Type::new_arr(Type::new_none(), Type::new_none()),
                    Expression::new_binop(Expression::new_ident("w"), BinOp::Add, Expression::new_ident("y")))),
        ),
    );
    println!("{:?}", &prog);
    eval(prog);
}
