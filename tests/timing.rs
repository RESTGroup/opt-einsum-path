use opt_einsum_path::contract::contract_path;

#[test]
fn test_large_greedy() {
    let einsum_str = "ab,ac,dcf,dbe,gfi,geh,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÁ,ZXÀ,ÂÁÄ,ÂÀÃ,ÅÄÇ,ÅÃÆ,ÈÇÊ,ÈÆÉ,ËÊÍ,ËÉÌ,ÎÍÐ,ÎÌÏ,ÑÐÓ,ÑÏÒ,ÔÓÖ,ÔÒÕ,×ÖÙ,×ÕØ,ÚÙÜ,ÚØÛ,ÝÜß,ÝÛÞ,àßâ,àÞá,ãâ,ãá";
    let len_ops = einsum_str.split(',').count();
    let shapes = vec![vec![3, 10], vec![3, 10]]
        .into_iter()
        .chain(std::iter::repeat_n(vec![3, 10, 10], len_ops - 4))
        .chain(vec![vec![3, 10], vec![3, 10]])
        .collect::<Vec<_>>();
    let time = std::time::Instant::now();
    let (_, info) = contract_path(einsum_str, &shapes, true, "dp", None).unwrap();
    println!("Elapsed time: {:?}", time.elapsed());
    println!("{:?}", info.opt_cost);
}
