#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(arbitrary_self_types)]
#![allow(non_upper_case_globals)]
#![feature(hash_set_entry)]

use calcium::Number;
use core::panic;
use gc::{unsafe_empty_trace, Finalize, Gc, GcCell, Trace};
use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{HashMap, HashSet},
    hash::Hash,
};

macro_rules! simple_empty_finalize_trace {
    ($($T:ty),*) => {
        $(
            impl Finalize for $T {}
            unsafe impl Trace for $T { unsafe_empty_trace!(); }
        )*
    }
}

#[derive(Debug, Trace, Finalize)]
enum Expression {
    Number(calcium::Number),
    Bool(bool),
    String(String),
    BinOp(Gc<Expression>, BinOp, Gc<Expression>),
    Ident(String),
    Let(Gc<Expression>, Gc<Expression>, Gc<Expression>),
    If(Gc<Expression>, Gc<Expression>, Gc<Expression>),
    Tuple(Vec<(Option<Gc<Expression>>, Gc<Expression>)>),
    List(Vec<Gc<Expression>>),
    Selector(Gc<Expression>, Gc<Expression>),
    Index(Gc<Expression>, Gc<Expression>),
    Lam(Gc<Expression>, Gc<Type>, Gc<Expression>), // add ident for parameter so that it always looks it up in the context? is the param type necessary?
    Ctl(
        Gc<Expression>, /* The lambda/closure */
        usize,          /* the index of the starting frame of the delimited continuation */
    ), // an effect handler
    Del(Vec<Frame>),                               // a delimited continuation
    Closure(
        Vec<(String, GcCell<Option<Gc<Expression>>>)>,
        Gc<Expression>,
    ),
    App(Gc<Expression>, Gc<Expression>), // Reuse for constructors?
    Fix(Gc<Expression>),
    Case(Gc<Expression>, Vec<(String, String, Gc<Expression>)>), // Compile exhaustive patterns to this, since all we need is an eliminator.
                                                                 // TODO: Effects
}

#[derive(Debug, Trace, Finalize)]
enum Type {
    None,
    // TODO
    // tuples, lists, string, bool, number, abs, variant, id
}

impl Expression {
    fn is_val(&self) -> bool {
        todo!()
    }
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
    fn new_tuple(v: Vec<(Option<Gc<Expression>>, Gc<Expression>)>) -> Gc<Expression> {
        Expression::Tuple(v).into()
    }
    fn new_list(v: Vec<Gc<Expression>>) -> Gc<Expression> {
        Expression::List(v).into()
    }
    fn new_selector(x: Gc<Expression>, field: Gc<Expression>) -> Gc<Expression> {
        Expression::Selector(x, field).into()
    }
    fn new_index(x: Gc<Expression>, i: Gc<Expression>) -> Gc<Expression> {
        Expression::Index(x, i).into()
    }
    fn new_lam(bind: Gc<Expression>, ty: Gc<Type>, body: Gc<Expression>) -> Gc<Expression> {
        Expression::Lam(bind, ty, body).into()
    }
    fn new_sulam(bind: &str, body: Gc<Expression>) -> Gc<Expression> {
        Expression::new_lam(Expression::new_ident(bind), Type::None.into(), body)
    }
    fn new_app(fun: Gc<Expression>, arg: Gc<Expression>) -> Gc<Expression> {
        Expression::App(fun, arg).into()
    }
    fn new_closure(
        env: Vec<(String, GcCell<Option<Gc<Expression>>>)>,
        lam: Gc<Expression>,
    ) -> Gc<Expression> {
        Expression::Closure(env, lam).into()
    }
    fn new_fix(x: Gc<Expression>) -> Gc<Expression> {
        Expression::Fix(x).into()
    }

    // fn new_ctl(fresh_name_resume: Gc<Expression>, fresh_name_bind: Gc<Expression>, ty: Gc<Type>, body: Gc<Expression>, oi: Option<usize>) -> Gc<Expression> {
    //     Expression::Ctl(fresh_name_resume, fresh_name_bind, ty, body, oi).into()
    // }

    fn new_ctl(lam: Gc<Expression>, frame_index: usize) -> Gc<Expression> {
        Expression::Ctl(lam, frame_index).into()
    }

    fn new_del(del: Vec<Frame>) -> Gc<Expression> {
        Expression::Del(del).into()
    }

    // fn new_ctl(
    //     bind: Gc<Expression>,
    //     ty: Gc<Type>,
    //     body: Gc<Expression>,
    //     frame_index: Option<usize>,
    // ) -> Gc<Expression> {
    //     Expression::Ctl(bind, ty, body, frame_index).into()
    // }
    // fn new_del(del: Vec<Frame>) -> Gc<Expression> {
    //     Expression::Del(del).into()
    // }
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    // Assign,
}

simple_empty_finalize_trace![BinOp];

// struct Interpreter {
//     context: HashMap<String, >
// }

// struct Environment {
//     context: HashMap<String, Option<Gc<Expression>>>,
//     parent: Option<Gc<Environment>>,
// }

// impl Environment {
//     fn lookup(&self, name: &String) -> Gc<Expression> {
//         match &self.context[name] {
//             Some(v) => v.clone(),
//             None => match &self.parent {
//                 Some(p) => p.lookup(name),
//                 None => panic!("couldn't find identifier"),
//             },
//         }
//     }
//     fn insert(&mut self, name: String, x: Gc<Expression>) {
//         self.context.insert(name, Some(x));
//         // self.context[name] = Some(x);
//     }
// }

// struct ContextEntry {
//     name: String,
//     x: Gc<Expression>,
// }

#[derive(Trace, Finalize)]
struct Context {
    // stored at front (prepended)
    parent: Option<Gc<Context>>,
    name: String,
    x: Gc<Expression>,
}

impl Context {
    fn lookup(self: Gc<Context>, name: &String) -> Gc<Expression> {
        if &self.name == name {
            self.x.clone()
        } else if let Some(p) = self.parent.clone() {
            p.lookup(name)
        } else {
            panic!("couldn't find identifier")
        }
    }
    fn with(self: Gc<Context>, name: String, x: &Gc<Expression>) -> Gc<Context> {
        Context::new(Some(self), name, x)
    }
    fn new(parent: Option<Gc<Context>>, name: String, x: &Gc<Expression>) -> Gc<Context> {
        Context {
            parent: parent,
            name: name,
            x: x.clone(),
        }
        .into()
    }
}

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
        Some(Frame::Rec { expr, locals, rv }) => rv.insert(v),
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

fn resolve_names(
    m: &mut Vec<(String, String)>,
    free_vars: &mut Vec<(String, GcCell<Option<Gc<Expression>>>)>,
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
            unique_names.remove(&fresh_name); // TODO: is this necessary? don't we want globally unique names?
            Expression::new_let(Expression::new_ident(&fresh_name), rhs, body)
        }
        Expression::If(cond, then, els) => {
            let cond = resolve_names(m, free_vars, unique_names, dist, cond);
            let then = resolve_names(m, free_vars, unique_names, dist, then);
            let els = resolve_names(m, free_vars, unique_names, dist, els);
            Expression::new_if(cond, then, els)
        }
        Expression::Tuple(_) => todo!(),
        Expression::List(_) => todo!(),
        Expression::Selector(_, _) => todo!(),
        Expression::Index(_, _) => todo!(),
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
                free_vars.push((s.to_string(), GcCell::new(None)));
            }
            return Expression::new_ident(&s);
        }
        Expression::Lam(bind, /*TODO: resolve names in types */ ty, body) => {
            let Expression::Ident(x) = &**bind else { panic!("expected identifier in lambda binding") };
            let fresh_name = add_fresh_name(unique_names, x);
            m.push((x.clone(), fresh_name.clone()));
            let mut free_vars = Vec::new();
            let body = resolve_names(m, &mut free_vars, unique_names, 0, body);
            m.pop();
            unique_names.remove(&fresh_name); // TODO: is this necessary? don't we want globally unique names?
            Expression::new_closure(
                free_vars,
                Expression::new_lam(Expression::new_ident(&fresh_name), ty.clone(), body),
            )
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
            Expression::new_ctl(resolve_names(m, free_vars, unique_names, dist, lam), *i)
        }
        // Expression::Ctl(resume_bind, param_bind, ty, body, oi) => { // TODO when we add tuples, make resume a field in the tuple
        //     // since this is not a closure, we don't need to handle free vars
        //     // treat this like Let
        //     let Expression::Ident(resume_name) = &**resume_bind else { panic!("expected identifier in handler binding") };
        //     let Expression::Ident(param_name) = &**param_bind else { panic!("expected identifier in handler binding") };
        //     let fresh_name_resume = add_fresh_name(unique_names, resume_name);
        //     let fresh_name_bind = add_fresh_name(unique_names, param_name);
        //     m.push((resume_name.clone(), fresh_name_resume.clone()));
        //     m.push((param_name.clone(), fresh_name_bind.clone()));
        //     let body = resolve_names(m, free_vars, unique_names, dist+2, body);
        //     m.pop();
        //     m.pop();
        //     unique_names.remove(&fresh_name_resume); // TODO: is this necessary? don't we want globally unique names?
        //     unique_names.remove(&fresh_name_bind); // TODO: is this necessary? don't we want globally unique names?
        //     Expression::new_ctl(Expression::new_ident(&fresh_name_resume), Expression::new_ident(&fresh_name_bind), ty.clone(), body, *oi)
        // },
        _ => prog.clone(),
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
                Expression::Closure(env, abs) => {
                    // for each free variable, look it up in the environment, and update the GcCell
                    for (name, cell) in env {
                        let val = lookup_id(&stack, name);
                        _ = cell.borrow_mut().insert(val);
                    }
                    to_set = Some(expr.clone())
                },
                Expression::Ident(s) => {
                    let val = lookup_id(&stack, s);
                    to_set = Some(val)
                },
                Expression::BinOp(l, op, r) => match rv {
                    Some(v) => {
                        let mut locals = locals.borrow_mut();
                        if let Some(v1) = locals.get("1") {
                            let res = match (&**v1, op, &**v) {
                                (Expression::Number(l), BinOp::Add, Expression::Number(r)) => Expression::new_number(l+r),
                                (Expression::Number(l), BinOp::Sub, Expression::Number(r)) => Expression::new_number(l-r),
                                (Expression::Number(l), BinOp::Mul, Expression::Number(r)) => Expression::new_number(l*r),
                                (Expression::Number(l), BinOp::Div, Expression::Number(r)) => Expression::new_number(l/r),
                                _ => panic!("left and right were not numbers"),
                            };
                            to_set = Some(res);
                        } else {
                            locals.insert("1".to_string(), v.clone());
                            to_push = Some(Frame::new_rec(r.clone()));
                        }
                    },
                    None => to_push = Some(Frame::new_rec(l.clone())),
                },
                Expression::Let(id, x, body) if let Expression::Ident(id) = &**id => match rv {
                    Some(v) => {
                        let mut locals = locals.borrow_mut();
                        if locals.contains_key("1") {
                            to_set = Some(v.clone());
                        } else {
                            let mut res = v.clone();
                            if let Expression::Ctl(cls, 0) = &**v {
                                // if RHS is a Ctl, then we set its delimiter
                                // This is like the reset of a shift/reset.
                                // By constructing a handler, we set a delimiter on the stack.
                                // Our approach right now will be to copy the part of the stack.
                                // The delimiter will be that of the body, so it will be 1+the index of the current frame.
                                res = Expression::new_ctl(cls.clone(), stack.len());
                            }
                            locals.insert("1".to_string(), res.clone());
                            to_push = Some(Frame::new_rec_locals(body.clone(), [(id.clone(), res)]));
                        }
                    },
                    None => to_push = Some(Frame::new_rec(x.clone())),
                },
                Expression::Let(_, _, _) => todo!(),
                Expression::If(cond, then, els) => match rv {
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
                    None => to_push = Some(Frame::new_rec(cond.clone())),
                },
                Expression::Tuple(_) => todo!(),
                Expression::List(_) => todo!(),
                Expression::Selector(_, _) => todo!(),
                Expression::Index(_, _) => todo!(),
                Expression::Lam(_, _, _) => todo!(), // subsumed by closure
                Expression::App(fun, arg) =>{
                    // println!("{:?}", fun);
                    match rv {
                    // An App where fun is a continuation would require
                    // pushing the stack frames of k
                    // Are we sure that resume from the continuation isn't referenced accidentally?
                    Some(rv) => {
                        if locals.borrow().contains_key("2") {
                            // return body
                            to_set = Some(rv.clone());
                        } else if locals.borrow().contains_key("1") {
                            let v1 : Gc<Expression> = {
                                let mut locals = locals.borrow_mut();
                                locals.insert("2".to_string(), rv.clone());
                                locals.remove("1").unwrap()
                            };
                            match &*v1 {
                                Expression::Del(_) => to_restore = Some(v1.clone()),
                                Expression::Ctl(cls, frame_index) => {
                                    // println!("{:?} and frame_index={}, len={}", cls, *frame_index, stack.len());
                                    to_capture = Some(v1)
                                },
                                _ => {
                                    let Expression::Closure(env, abs) = &*v1 else { panic!("expected closure") };
                                    let Expression::Lam(bind, _, body) = &**abs else { panic!("expected abs") };
                                    let Expression::Ident(bind) = &**bind else { panic!("expected ident") };
                                    let mut fn_locals = HashMap::new();
                                    for (id, val) in env {
                                        let val = val.borrow().as_ref().unwrap().clone();
                                        fn_locals.insert(id.to_string(), val);
                                    }
                                    fn_locals.insert(bind.to_string(), rv.clone());
                                    to_push = Some(Frame::new_rec_hm(body.clone(), fn_locals));
                                }
                            }
                        } else {
                            // eval arg
                            locals.borrow_mut().insert("1".to_string(), rv.clone());
                            to_push = Some(Frame::new_rec(arg.clone()));
                        }
                    },
                    // eval fun
                    None => to_push = Some(Frame::new_rec(fun.clone())),
                }
            },
                Expression::Fix(x) => match rv {
                    Some(rv) => {
                        // let mut locals = locals.borrow_mut();
                        if locals.borrow().contains_key("1") {
                            to_set = Some(rv.clone());
                        } else {
                            {
                                locals.borrow_mut().insert("1".to_string(), rv.clone());
                            }
                            let Expression::Closure(env, abs) = &**rv else { panic!("expected closure") };
                            let Expression::Lam(bind, _, body) = &**abs else { panic!("expected abs") };
                            let Expression::Ident(bind) = &**bind else { panic!("expected ident") };
                            let mut fn_locals = HashMap::new();
                            for (id, val) in env {
                                let val = val.borrow().as_ref().unwrap().clone();
                                fn_locals.insert(id.to_string(), val);
                            }
                            let arg = Expression::new_fix(rv.clone());
                            fn_locals.insert(bind.to_string(), arg);
                            to_push = Some(Frame::new_rec_hm(body.clone(), fn_locals));
                        }
                    },
                    None => to_push = Some(Frame::new_rec(x.clone())),
                },
                Expression::Case(_, _) => todo!(),
                // Expression::Ctl(_, _, _, _, _) => todo!(),
                Expression::Ctl(fun, frame_index) => match rv {
                    // Evaluate fun to a closure
                    // Return a Ctl expression with the closure and the frame index
                    Some(cls) => {
                        let Expression::Closure(_, _) = &**cls else { panic!("expected closure") };
                        to_set = Some(Expression::new_ctl(cls.clone(), *frame_index))
                    },
                    None => to_push = Some(Frame::new_rec(fun.clone())),
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
            if let Some(Frame::Rec { expr, locals, rv }) = stack.last_mut() {
                rv.insert(arg);
            }
        }
        if let Some(Expression::Ctl(cls, del)) = to_capture.as_deref() {
            let arg = {
                let Some(Frame::Rec { expr: _, locals, rv: _ }) = stack.last() else { panic!("expected record")};
                locals.borrow().get("2").unwrap().clone()
            };
            if *del >= stack.len() {
                panic!("handler out of scope"); // TODO: make this an effect as well
            }
            let k : Vec<Frame> = stack.drain(del..).collect();
            let next_exp = Expression::new_app(Expression::new_app(cls.clone(), arg), Expression::new_del(k)); // do we want cls arg k or cls k arg?
            // println!("pushing cls arg k");
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

fn eval_stack(x: Gc<Expression>) -> Gc<Expression> {
    eval_with_stack(vec![Frame::Rv(None), Frame::new_rec(x)])
}

fn eval(ctx: &Option<Gc<Context>>, x: &Gc<Expression>) -> Gc<Expression> {
    let (ctx, x) = (ctx.clone(), x.clone());
    match &*x {
        Expression::Number(_) => x,
        Expression::Bool(_) => x,
        Expression::String(_) => x,
        Expression::BinOp(left, op, right) => match (&*eval(&ctx, left), op, &*eval(&ctx, right)) {
            (Expression::Number(l), BinOp::Add, Expression::Number(r)) => Expression::new_number(l+r),
            (Expression::Number(l), BinOp::Sub, Expression::Number(r)) => Expression::new_number(l-r),
            (Expression::Number(l), BinOp::Mul, Expression::Number(r)) => Expression::new_number(l*r),
            (Expression::Number(l), BinOp::Div, Expression::Number(r)) => Expression::new_number(l/r),
            _ => panic!("left and right were not numbers")
        },
        Expression::Ident(id) if let Some(ctx) = ctx => ctx.lookup(id),
        Expression::Let(id, v, x) if let Expression::Ident(id) = &**id => match ctx.clone() {
            Some(ctx) => eval(&Some(ctx.with(id.to_string(), v)), x),
            None => eval(&Some(Context::new(None, id.to_string(), v)), x),
        },
        Expression::If(cond, then, els) if let Expression::Bool(cond) = *eval(&ctx, cond) =>
            if cond { eval(&ctx, then) } else { eval(&ctx, els) },
        Expression::Tuple(elements) => Expression::new_tuple(elements.iter().map(
            |elem| (elem.0.clone(), eval(&ctx, &elem.1))
        ).collect()),
        Expression::List(elements) => Expression::new_list(elements.iter().map(|elem| eval(&ctx, elem)).collect()),
        Expression::Selector(x, sel) if let Expression::Ident(id) = &**sel =>
        if let Expression::Tuple(r) = &*eval(&ctx, x) {
            r.iter().find_map(|ele|
                if let Some(x) = &ele.0 && let Expression::Ident(got)= &**x && id == got {
                    Some(ele.1.clone())
                } else {
                    None
                }).unwrap()
        } else {
            panic!("selector expression wasn't made up of tuple and ident")
        },
        Expression::Index(x, i) if let Expression::Number(i) = &*eval(&ctx, i) => match &*eval(&ctx, x) {
            Expression::Tuple(r) => r[Into::<usize>::into(i)].1.clone(),
            Expression::List(l) => l[Into::<usize>::into(i)].clone(),
            _ => panic!("expression is not indexable"),
        }
        Expression::Lam(_, _, _) => x,
        Expression::App(f, arg) => match &*eval(&ctx, f) {
            Expression::Lam(param, _param_ty, body) => {
                let arg = eval(&ctx, arg);
                if !arg.is_val() {
                    todo!()
                }
                if let Expression::Ident(id) = &**param {
                    match ctx.clone() {
                        Some(ctx) => eval(&Some(ctx.with(id.to_string(), &arg)), body),
                        None => eval(&Some(Context::new(None, id.to_string(), &arg)), body),
                    }
                } else {
                    todo!()
                }
            },
            _ => todo!(),
            // TODO: handle constructor. is it an ident we look up in some scope?
        },
        Expression::Fix(x)  if let Expression::Lam(param, _, body) = &*eval(&ctx, x) =>
        if let Expression::Ident(id) = &**param {
            match ctx.clone() {
                Some(ctx) => eval(&Some(ctx.with(id.to_string(), body)), body),
                None => eval(&Some(Context::new(None, id.to_string(), body)), body),
            }
        } else {
            todo!()
        },
        _ => panic!("unexpected expression"),
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


    println!(r#"evaluating
    let x = ctl(fun arg k -> k(arg))
    in x(1) + 2
    "#);
    let prog = Expression::new_let(
        Expression::new_ident("x"),
        Expression::new_ctl(Expression::new_lam(Expression::new_ident("arg"), Type::None.into(), Expression::new_lam(Expression::new_ident("k"), Type::None.into(), Expression::new_app(Expression::new_ident("k"), Expression::new_ident("arg")))), 0),
        Expression::new_binop(Expression::new_app(Expression::new_ident("x"), Expression::new_number(Number::one())), BinOp::Add, Expression::new_number(Number::from_ulong(2)))
    );

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
    //         Expression::new_ctl(Expression::new_sulam("arg", Expression::new_sulam("k", Expression::new_ident("arg"))), 0),
    //         Expression::new_ident("y"),
    //     ),
    //     Expression::new_app(Expression::new_ident("x"), Expression::new_int(1))
    // );

    println!("{:?}", &prog);
    let mut m = Vec::new();
    let mut free_vars = Vec::new();
    let mut unique_names = HashSet::new();
    let prog = resolve_names(&mut m, &mut free_vars, &mut unique_names, 0, &prog);
    println!("{:?}", &prog);
    let stackres = eval_stack(prog);
    println!("stack res = {:?}", stackres);
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
}
